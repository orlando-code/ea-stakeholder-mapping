"""Person recommendation system based on semantic similarity.

This module provides a PersonRecommender class that suggests connections
between attendees based on their profiles (expertise, interests, biography).

Recommendation types:
1. Similar - High similarity for direct collaboration
2. Complementary - Moderate similarity, different focus areas
3. Skill-match - Their expertise matches your interests (and vice versa)
4. Wildcard - Maximally different for unexpected connections
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Lazy imports to avoid dependency issues
SentenceTransformer = None
cosine_similarity = None


def _ensure_dependencies():
    """Lazily import dependencies."""
    global SentenceTransformer, cosine_similarity
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        from sklearn.metrics.pairwise import cosine_similarity as cs

        SentenceTransformer = ST
        cosine_similarity = cs


@dataclass
class Recommendation:
    """A single person recommendation with explanation."""

    person_idx: int
    score: float
    recommendation_type: str  # "similar", "complementary", "skill_match", "wildcard"
    explanation: str
    shared_topics: list[str] = field(default_factory=list)
    complementary_topics: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Recommendation(idx={self.person_idx}, type={self.recommendation_type}, score={self.score:.3f})"


@dataclass
class PersonRecommendations:
    """All recommendations for a single person."""

    person_idx: int
    person_name: Optional[str]
    similar: list[Recommendation] = field(default_factory=list)
    complementary: list[Recommendation] = field(default_factory=list)
    skill_match: list[Recommendation] = field(default_factory=list)
    wildcard: list[Recommendation] = field(default_factory=list)

    def all_recommendations(self) -> list[Recommendation]:
        """Get all recommendations as a flat list."""
        return self.similar + self.complementary + self.skill_match + self.wildcard

    def to_dataframe(self) -> pd.DataFrame:
        """Convert recommendations to DataFrame."""
        rows = []
        for rec in self.all_recommendations():
            rows.append(
                {
                    "person_idx": rec.person_idx,
                    "type": rec.recommendation_type,
                    "score": rec.score,
                    "explanation": rec.explanation,
                    "shared_topics": ", ".join(rec.shared_topics),
                    "complementary_topics": ", ".join(rec.complementary_topics),
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Get human-readable summary of recommendations."""
        lines = [f"Recommendations for '{self.person_idx}'"]
        if self.person_name:
            lines[0] += f" ({self.person_name})"
        lines.append("=" * 50)

        for category, recs in [
            ("🤝 Similar (collaboration – high profile similarity)", self.similar),
            (
                "🔄 Complementary (cross-pollination – moderate similarity, different focus areas)",
                self.complementary,
            ),
            ("🎯 Skill Match (expertise ↔ interests)", self.skill_match),
            ("🎲 Wildcard (unexpected – maximally different)", self.wildcard),
        ]:
            if recs:
                lines.append(f"\n{category}:")
                for rec in recs:
                    lines.append(f"  • Person {rec.person_idx} ({rec.score:.0%})")
                    lines.append(f"      {rec.explanation}")

        return "\n".join(lines)


class PersonRecommender:
    """Recommend connections between people based on semantic similarity.

    Uses sentence transformers to embed each person's profile (expertise,
    interests, biography) and computes various similarity metrics to
    suggest different types of connections.

    Example:
        recommender = PersonRecommender(df)
        recs = recommender.recommend(person_idx=0, top_k=3)
        print(recs.summary())

        # Or get all recommendations as a DataFrame
        df_recs = recs.to_dataframe()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        expertise_col: str = "expertise_parsed",
        interests_col: str = "interests_parsed",
        bio_col: str = "biography",
        name_col: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2",
        augment_with_extraction: Optional[str] = None,
    ):
        """Initialize recommender with attendee data.

        Args:
            df: DataFrame with attendee data
            expertise_col: Column containing expertise (list of strings)
            interests_col: Column containing interests (list of strings)
            bio_col: Column containing biography text
            name_col: Optional column with person names (for display)
            model_name: Sentence transformer model to use
            augment_with_extraction: Optionally augment expertise/interests with
                extracted cause areas. Options: "llm" (use llm_cause_areas),
                "nlp" (use nlp_cause_areas), or None (don't augment)
        """
        _ensure_dependencies()

        self.df = df.reset_index(drop=True)
        self.expertise_col = expertise_col
        self.interests_col = interests_col
        self.bio_col = bio_col
        self.name_col = name_col
        self.augment_with_extraction = augment_with_extraction
        self.model = SentenceTransformer(model_name)

        # Track indices of people with empty profiles (to exclude from recommendations)
        self.empty_profile_indices: set[int] = set()

        # Compute embeddings
        self._compute_embeddings()

    def _get_list_field(self, row, col: str) -> list[str]:
        """Safely get list field from row."""
        if col not in self.df.columns:
            return []
        val = row.get(col)
        if isinstance(val, list):
            return [str(v).lower().strip() for v in val if v]
        elif isinstance(val, str) and val:
            return [val.lower().strip()]
        return []

    def _get_text_field(self, row, col: str) -> str:
        """Safely get text field from row."""
        if col not in self.df.columns:
            return ""
        val = row.get(col)
        if pd.isna(val):
            return ""
        return str(val)

    def _get_extraction_col(self) -> Optional[str]:
        """Get the extraction column name based on augment_with_extraction setting."""
        if self.augment_with_extraction == "llm":
            return "llm_cause_areas"
        elif self.augment_with_extraction == "nlp":
            return "nlp_cause_areas"
        return None

    def _compute_embeddings(self):
        """Compute embeddings for each person's profile."""
        print("Computing person embeddings...")

        # Determine extraction column for augmentation
        extraction_col = self._get_extraction_col()
        if extraction_col and extraction_col in self.df.columns:
            print(f"  Augmenting with {self.augment_with_extraction.upper()} cause areas")

        # Store parsed fields for later use
        self.expertise_lists = []
        self.interests_lists = []
        profiles = []

        for idx, row in self.df.iterrows():
            expertise = self._get_list_field(row, self.expertise_col)
            interests = self._get_list_field(row, self.interests_col)
            bio = self._get_text_field(row, self.bio_col)

            # Augment with extracted cause areas if requested
            if extraction_col and extraction_col in self.df.columns:
                extracted = self._get_list_field(row, extraction_col)
                # Add extracted items to both expertise and interests (as general topics)
                # Use set to avoid duplicates
                expertise = list(set(expertise) | set(extracted))
                interests = list(set(interests) | set(extracted))

            self.expertise_lists.append(expertise)
            self.interests_lists.append(interests)

            # Combine into profile text
            parts = []
            if expertise:
                parts.append("Expertise: " + ", ".join(expertise))
            if interests:
                parts.append("Interests: " + ", ".join(interests))
            if bio:
                parts.append(bio)

            profile_text = " ".join(parts) if parts else ""
            profiles.append(profile_text)

            # Track empty profiles
            if not profile_text.strip():
                self.empty_profile_indices.add(idx)

        # Report empty profiles
        n_empty = len(self.empty_profile_indices)
        if n_empty > 0:
            print(f"  Found {n_empty} people with empty profiles (will be excluded)")

        # Compute full profile embeddings
        self.profile_embeddings = self.model.encode(profiles, show_progress_bar=True)
        self.profile_similarity = cosine_similarity(self.profile_embeddings)

        # Compute separate expertise and interest embeddings for skill matching
        expertise_texts = [" ".join(e) if e else "" for e in self.expertise_lists]
        interest_texts = [" ".join(i) if i else "" for i in self.interests_lists]

        self.expertise_embeddings = self.model.encode(expertise_texts, show_progress_bar=False)
        self.interests_embeddings = self.model.encode(interest_texts, show_progress_bar=False)

        # Cross-similarity: how well does person A's expertise match person B's interests?
        self.expertise_to_interests = cosine_similarity(
            self.expertise_embeddings, self.interests_embeddings
        )

        n_valid = len(self.df) - n_empty
        print(
            f"Computed embeddings for {n_valid} people ({n_empty} excluded due to empty profiles)"
        )

    def _find_shared_topics(self, idx1: int, idx2: int) -> list[str]:
        """Find topics shared between two people."""
        set1 = set(self.expertise_lists[idx1] + self.interests_lists[idx1])
        set2 = set(self.expertise_lists[idx2] + self.interests_lists[idx2])
        return list(set1 & set2)[:5]  # Top 5

    def _find_complementary_topics(self, idx1: int, idx2: int) -> tuple[list[str], list[str]]:
        """Find topics unique to each person."""
        set1 = set(self.expertise_lists[idx1] + self.interests_lists[idx1])
        set2 = set(self.expertise_lists[idx2] + self.interests_lists[idx2])
        unique1 = list(set1 - set2)[:3]
        unique2 = list(set2 - set1)[:3]
        return unique1, unique2

    def _find_skill_match_topics(self, idx1: int, idx2: int) -> tuple[list[str], list[str]]:
        """Find where person1's expertise matches person2's interests and vice versa."""
        exp1 = set(self.expertise_lists[idx1])
        int1 = set(self.interests_lists[idx1])
        exp2 = set(self.expertise_lists[idx2])
        int2 = set(self.interests_lists[idx2])

        # Person 1's expertise matching person 2's interests
        exp1_to_int2 = list(exp1 & int2)[:3]
        # Person 2's expertise matching person 1's interests
        exp2_to_int1 = list(exp2 & int1)[:3]

        return exp1_to_int2, exp2_to_int1

    def _get_person_name(self, idx: int) -> Optional[str]:
        """Get person's name if available."""
        if self.name_col and self.name_col in self.df.columns:
            return self.df.iloc[idx][self.name_col]
        return None

    def recommend_similar(
        self, person_idx: int, top_k: int = 5, exclude_indices: set[int] = None
    ) -> list[Recommendation]:
        """Find people most similar to given person (for direct collaboration).

        Args:
            person_idx: Index of person to find matches for
            top_k: Number of recommendations
            exclude_indices: Indices to exclude from results
        Returns:
            List of Recommendation objects
        """
        exclude = (exclude_indices or set()) | self.empty_profile_indices
        exclude.add(person_idx)

        sims = self.profile_similarity[person_idx].copy()
        sims[list(exclude)] = -1

        indices = np.argsort(sims)[::-1][:top_k]

        recommendations = []
        for idx in indices:
            if sims[idx] < 0:
                continue

            shared = self._find_shared_topics(person_idx, idx)
            # explanation = f"({sims[idx]:.0%})."
            # if shared:
            explanation = f"Shared interests: {', '.join(shared[:3])}"

            recommendations.append(
                Recommendation(
                    person_idx=int(idx),
                    score=float(sims[idx]),
                    recommendation_type="similar",
                    explanation=explanation,
                    shared_topics=shared,
                )
            )

        return recommendations

    def recommend_complementary(
        self,
        person_idx: int,
        top_k: int = 5,
        exclude_indices: set[int] = None,
        similarity_range: tuple[float, float] = (0.3, 0.65),
    ) -> list[Recommendation]:
        """Find people with moderate similarity (for cross-pollination).

        These are people who share some common ground but have different
        focus areas - good for learning from different perspectives.

        Args:
            person_idx: Index of person to find matches for
            top_k: Number of recommendations
            exclude_indices: Indices to exclude from results
            similarity_range: (min, max) similarity range to target
        Returns:
            List of Recommendation objects
        """
        exclude = (exclude_indices or set()) | self.empty_profile_indices
        exclude.add(person_idx)

        sims = self.profile_similarity[person_idx].copy()
        min_sim, max_sim = similarity_range

        # Score based on how close to middle of range
        target = (min_sim + max_sim) / 2
        scores = np.zeros_like(sims)

        for i, sim in enumerate(sims):
            if i in exclude:
                scores[i] = -1
            elif min_sim <= sim <= max_sim:
                # Higher score for being in range, bonus for being near target
                scores[i] = 1.0 - abs(sim - target) / (max_sim - min_sim)
            else:
                scores[i] = -1

        indices = np.argsort(scores)[::-1][:top_k]

        recommendations = []
        for idx in indices:
            if scores[idx] < 0:
                continue

            sim = sims[idx]
            unique_you, unique_them = self._find_complementary_topics(person_idx, idx)
            shared = self._find_shared_topics(person_idx, idx)

            explanation = ""
            if shared:
                explanation += f"Common ground: {', '.join(shared[:2])}."
            if unique_them:
                explanation += f" They bring: {', '.join(unique_them[:2])}"

            recommendations.append(
                Recommendation(
                    person_idx=int(idx),
                    score=float(sim),
                    recommendation_type="complementary",
                    explanation=explanation,
                    shared_topics=shared,
                    complementary_topics=unique_them,
                )
            )

        return recommendations

    def recommend_skill_match(
        self, person_idx: int, top_k: int = 5, exclude_indices: set[int] = None
    ) -> list[Recommendation]:
        """Find people whose expertise matches your interests (and vice versa).

        These are people who can teach you what you want to learn, and/or
        who want to learn what you know.

        Args:
            person_idx: Index of person to find matches for
            top_k: Number of recommendations
            exclude_indices: Indices to exclude from results
        Returns:
            List of Recommendation objects
        """
        exclude = (exclude_indices or set()) | self.empty_profile_indices
        exclude.add(person_idx)

        # Their expertise → my interests
        their_exp_my_int = self.expertise_to_interests[:, person_idx].copy()
        # My expertise → their interests
        my_exp_their_int = self.expertise_to_interests[person_idx, :].copy()

        # Combined score: average of both directions
        scores = (their_exp_my_int + my_exp_their_int) / 2
        scores[list(exclude)] = -1

        indices = np.argsort(scores)[::-1][:top_k]

        recommendations = []
        for idx in indices:
            if scores[idx] < 0:
                continue

            exp1_to_int2, exp2_to_int1 = self._find_skill_match_topics(person_idx, idx)

            explanation_parts = []
            if exp2_to_int1:
                explanation_parts.append(
                    f"They have expertise in: {', '.join(exp2_to_int1)} (your interests)"
                )
            if exp1_to_int2:
                explanation_parts.append(
                    f"You have expertise in: {', '.join(exp1_to_int2)} (their interests)"
                )

            if not explanation_parts:
                # Fall back to embedding-based explanation
                explanation = f"Skill alignment score: {scores[idx]:.0%}"
            else:
                explanation = ". ".join(explanation_parts)

            recommendations.append(
                Recommendation(
                    person_idx=int(idx),
                    score=float(scores[idx]),
                    recommendation_type="skill_match",
                    explanation=explanation,
                    shared_topics=exp2_to_int1,  # What they can teach you
                    complementary_topics=exp1_to_int2,  # What you can teach them
                )
            )

        return recommendations

    def recommend_wildcard(
        self, person_idx: int, top_k: int = 3, exclude_indices: set[int] = None
    ) -> list[Recommendation]:
        """Find maximally different people (for unexpected connections).

        Sometimes the best connections come from completely different fields.

        Args:
            person_idx: Index of person to find matches for
            top_k: Number of recommendations
            exclude_indices: Indices to exclude from results
        Returns:
            List of Recommendation objects
        """
        exclude = (exclude_indices or set()) | self.empty_profile_indices
        exclude.add(person_idx)

        sims = self.profile_similarity[person_idx].copy()
        sims[list(exclude)] = 2  # Set excluded to high so they sort last

        # Get LEAST similar people (empty profiles already excluded)
        indices = np.argsort(sims)[:top_k]

        recommendations = []
        for idx in indices:
            if sims[idx] >= 2:  # Excluded
                continue

            sim = sims[idx]
            _, unique_them = self._find_complementary_topics(person_idx, idx)

            explanation = f"Very different profile ({sim:.0%} similarity)."
            if unique_them:
                explanation += f" Completely different focus: {', '.join(unique_them[:3])}"

            recommendations.append(
                Recommendation(
                    person_idx=int(idx),
                    score=float(1 - sim),  # Invert: lower similarity = higher score
                    recommendation_type="wildcard",
                    explanation=explanation,
                    complementary_topics=unique_them,
                )
            )

        return recommendations

    def recommend(
        self,
        person_idx: int,
        top_k: int = 5,
        include_types: Optional[list[str]] = None,
    ) -> PersonRecommendations:
        """Get all recommendation types for a person.

        Args:
            person_idx: Index of person to find matches for
            top_k: Number of recommendations per type
            include_types: Types to include (default: all)
                Options: "similar", "complementary", "skill_match", "wildcard"
        Returns:
            PersonRecommendations with all recommendation types
        """
        if include_types is None:
            include_types = ["similar", "complementary", "skill_match", "wildcard"]

        recs = PersonRecommendations(
            person_idx=person_idx,
            person_name=self._get_person_name(person_idx),
        )

        # Track all recommended indices to avoid duplicates across types
        recommended = set()

        if "similar" in include_types:
            recs.similar = self.recommend_similar(person_idx, top_k, recommended)
            recommended.update(r.person_idx for r in recs.similar)

        if "complementary" in include_types:
            recs.complementary = self.recommend_complementary(person_idx, top_k, recommended)
            recommended.update(r.person_idx for r in recs.complementary)

        if "skill_match" in include_types:
            recs.skill_match = self.recommend_skill_match(person_idx, top_k, recommended)
            recommended.update(r.person_idx for r in recs.skill_match)

        if "wildcard" in include_types:
            recs.wildcard = self.recommend_wildcard(person_idx, min(top_k, 3), recommended)

        return recs

    def recommend_for_all(
        self, top_k: int = 3, include_types: Optional[list[str]] = None
    ) -> dict[int, PersonRecommendations]:
        """Get recommendations for all people.

        Args:
            top_k: Number of recommendations per type per person
            include_types: Types to include (default: all)
        Returns:
            Dict mapping person_idx to PersonRecommendations
        """
        results = {}
        for idx in range(len(self.df)):
            results[idx] = self.recommend(idx, top_k, include_types)
        return results

    def find_best_matches(self, person_idx: int, top_k: int = 10) -> list[Recommendation]:
        """Get overall best matches across all recommendation types.

        Combines and ranks recommendations from all types.

        Args:
            person_idx: Index of person to find matches for
            top_k: Total number of recommendations to return
        Returns:
            List of Recommendation objects, sorted by score
        """
        recs = self.recommend(person_idx, top_k=top_k)
        all_recs = recs.all_recommendations()

        # Sort by score (descending)
        all_recs.sort(key=lambda r: r.score, reverse=True)

        return all_recs[:top_k]
