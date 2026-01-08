# EA Stakeholder Mapping

This repo fulfils the project component of an EA Project Fellowship with EA Cambridge. It aims to analyze EA conference attendee data: extract entities from biographies, visualize geographic distribution, cluster cause areas semantically, and recommend connections between attendees.

## README Structure
## Contents

- [EA Stakeholder Mapping](#ea-stakeholder-mapping)
  - [README Structure](#readme-structure)
  - [Contents](#contents)
  - [Features](#features)
  - [Results](#results)
  - [Installation](#installation)
    - [Ollama Setup (for LLM extraction)](#ollama-setup-for-llm-extraction)
  - [Quick Start](#quick-start)
  - [Pipeline API](#pipeline-api)
    - [Data Loading](#data-loading)
    - [Extraction](#extraction)
    - [Geographic Analysis](#geographic-analysis)
    - [Semantic Analysis](#semantic-analysis)
    - [Person Recommendations](#person-recommendations)
    - [LLM Majority Voting](#llm-majority-voting)
  - [Caching](#caching)
  - [EA Cause Categories](#ea-cause-categories)
  - [Architecture](#architecture)
  - [Configuration](#configuration)
  - [Development](#development)
  - [License](#license)
- [Project notes](#project-notes)
  - [Meta](#meta)
    - [Time commitment](#time-commitment)
    - [Use of LLMs](#use-of-llms)
    - [Advancement/Collaboration](#advancementcollaboration)
  - [Summary](#summary)
    - [General approach](#general-approach)
    - [Geographic extraction](#geographic-extraction)
    - [General NLP vs LLM](#general-nlp-vs-llm)
    - [Future work](#future-work)


## Features

- **Entity Extraction**: Extract locations, organizations, and cause areas from attendee biographies
  - **NLP Method**: Fast, deterministic extraction using spaCy NER and keyword matching
  - **LLM Method**: Accurate extraction using local Ollama with majority voting

- **Method Comparison**: Compare NLP vs LLM extraction performance (overlap, Jaccard similarity)

- **Geographic Visualization**: Interactive choropleth maps showing attendee distribution by country

- **Semantic Analysis**: Cluster cause areas by semantic similarity using sentence transformers
  - Network visualization showing cause area relationships
  - Automatic cluster labeling based on EA cause categories
  - Option to use predefined categories instead of unsupervised clustering

- **Person Recommendations**: Find connections between attendees based on:
  - Similar profiles (for collaboration)
  - Complementary interests (for cross-pollination)
  - Skill matching (their expertise = your interests)
  - Wildcards (maximally different profiles)


## Results

The following interactive graphs are the headline results of the codebase.

*What are the most-mentioned cause areas?* 
📈 [Cause area chart](https://orlando-code.github.io/ea-stakeholder-mapping/output/cause_area_chart.html)

*How are those cause areas clustered within the EA framework?*
🌳 [Cluster treemap](https://orlando-code.github.io/ea-stakeholder-mapping/output/cluster_treemap.html)

*How semantically similar are those cause areas?*
🕸️ [Interactive Semantic Network](https://orlando-code.github.io/ea-stakeholder-mapping/output/semantic_network.html)

*What are the geographical connections of attendees?*
🗺️ [Global map of attendee associations](https://orlando-code.github.io/ea-stakeholder-mapping/output/map.html)

*Who knows what, and who's interested in what?*
🧑‍🔬 [Expertise vs interest chart](https://orlando-code.github.io/ea-stakeholder-mapping/output/expertise_vs_interest_chart.html)

*Given this, what are some undervalued areas?*
💡 [Undervalued cause areas chart](https://orlando-code.github.io/ea-stakeholder-mapping/output/undervalued_chart.html)

*How do the NLP and LLM extraction methods compare?*
🤖 [NLP vs LLM keyword extraction comparison chart](https://orlando-code.github.io/ea-stakeholder-mapping/output/extraction_comparison_chart.html)

*How closely-associated are different cause areas?*
🔥 [Keyword similarity heatmap](https://orlando-code.github.io/ea-stakeholder-mapping/output/similarity_heatmap.html)

## Installation

```bash
# Clone and install
cd ea-stakeholder-mapping
pip install -e .

# Install spaCy model
python -m spacy download en_core_web_sm
# Or for better accuracy:
python -m spacy download en_core_web_trf

# Install semantic analysis dependencies (required for clustering & recommendations)
pip install sentence-transformers scikit-learn
```

### Ollama Setup (for LLM extraction)

A static snapshot of cached files are available in `.cache`. If you want to run this yourself – or on new data – you'll need to install your own local instance of Ollama: 

1. Install Ollama: https://ollama.ai/download
2. Start Ollama: `ollama serve`
3. Pull a model: `ollama pull llama3.2`

## Quick Start

```python
from sm import Pipeline

# Create pipeline with both extraction methods
pipe = Pipeline(methods=["nlp", "llm"])

# Load data
pipe.load_data("data/attendees.csv")

# Extract entities from text columns
pipe.extract(
    text_columns=["biography", "help_me"],
    semicolon_columns=["expertise", "interests"],
)

# Compare extraction methods
comparison = pipe.compare_methods()
print(comparison.summary())

# Analyze semantics with predefined EA categories
pipe.analyze_semantic(
    use_predefined_categories=True,
    min_category_size=3,  # Merge small categories
)

# Visualize
fig = pipe.create_semantic_network(min_mentions=2)
fig.show()
```

## Pipeline API

The `Pipeline` class provides a high-level interface for the complete analysis workflow.

### Data Loading

```python
from sm import Pipeline

pipe = Pipeline(methods=["nlp", "llm"])

# Load from file
pipe.load_data("data/attendees.csv", skip_rows=5, anonymize=True)

# Or set DataFrame directly
pipe.set_data(my_dataframe)
```

### Extraction

```python
# Extract from text columns with optional semicolon-separated columns
pipe.extract(
    text_columns=["biography", "help_me"],
    semicolon_columns=["expertise", "interests"],
    progress=True,
    parallel=True,  # Parallel LLM processing
)

# Adds columns: nlp_locations, nlp_organizations, nlp_cause_areas
#              llm_locations, llm_organizations, llm_cause_areas
#              expertise_parsed, interests_parsed
```

### Geographic Analysis

```python
# Analyze geographic distribution
country_counts, org_geo = pipe.analyze_geographic()

# Create interactive map
fig = pipe.create_map(show_organizations=True)
fig.show()
```

### Semantic Analysis

```python
# Unsupervised clustering (automatic)
pipe.analyze_semantic(
    n_clusters=None,  # Auto-determine
    min_mentions=2,
)

# Or use predefined EA categories
pipe.analyze_semantic(
    use_predefined_categories=True,
    similarity_threshold=0.3,  # Items below this go to "Other"
    min_category_size=3,       # Merge small categories
)

# Visualizations
pipe.create_semantic_network(min_mentions=2)
pipe.create_cause_area_chart(top_n=25)
pipe.create_expertise_vs_interest_chart(top_n=20)
pipe.create_undervalued_chart(top_n=20)  # High interest, low expertise
```

### Person Recommendations

```python
# Create recommender
recommender = pipe.create_recommender(
    augment_with_extraction="llm",  # Add LLM-extracted cause areas
)

# Get recommendations for a person
recs = recommender.recommend(person_idx=0, top_k=5)
print(recs.summary())

# Output:
# Recommendations for '0'
# ==================================================
# 
# 🤝 Similar (collaboration – high profile similarity):
#   • Person 12 (85%)
#       Shared interests: ai safety, governance, policy
# 
# 🔄 Complementary (cross-pollination – moderate similarity):
#   • Person 45 (52%)
#       Common ground: research. They bring: animal welfare
# 
# 🎯 Skill Match (expertise ↔ interests):
#   • Person 23 (71%)
#       They have expertise in: machine learning (your interests)
# 
# 🎲 Wildcard (unexpected – maximally different):
#   • Person 89 (82%)
#       Very different profile (18% similarity). Different focus: global health

# Get as DataFrame
df_recs = recs.to_dataframe()

# Get only specific types
recs = recommender.recommend(
    person_idx=0,
    include_types=["similar", "skill_match"]
)

# Best matches across all types
best = recommender.find_best_matches(person_idx=0, top_k=10)
```

### LLM Majority Voting

Meta's [Ollama3.2 (latest)](https://ollama.com/library/llama3.2) was used since it's open access and can be queried locally (good for privacy and keeping track of number of calls/energy use).

This model was released a year ago and packs a 128k context and 3.21B parameters into 2.0GB. Compared to more recent models e.g. [Olmo 3](https://ollama.com/library/olmo-3:latest) – sporting 32B parameters – this is comparatively lightweight!

To handle LLM stochasticity, extractions run multiple times and aggregate:

1. Run the same extraction prompt N times (default: 3)
2. Collect all extracted items across runs
3. Keep items appearing in >= 50% of runs

```python
from sm import LLMExtractor

# More runs = more stable results (but slower)
llm = LLMExtractor(
    n_runs=5,              # Run 5 times
    vote_threshold=0.6,    # Keep items in 60%+ of runs
)
```

## Caching

Results are cached to avoid recomputation:

```
.cache/
├── nlp/                    # spaCy extractions
│   ├── locations/
│   ├── organizations/
│   └── cause_areas/
├── llm/                    # LLM extractions
│   └── llama3.2/
│       ├── locations/
│       ├── organizations/
│       └── cause_areas/
└── geocoding/              # Geocoding results
```

```python
from sm import clear_cache, get_cache_stats

# View cache statistics
stats = get_cache_stats()
print(f"Cache: {stats['total_files']} files, {stats['total_size_mb']:.1f} MB")

# Clear specific category
clear_cache("llm")      # Clear only LLM cache
clear_cache("geocoding")

# Clear all
clear_cache()
```

## EA Cause Categories

Predefined EA cause area categories for clustering and labeling (stored in `config.py`):

- AI Safety & Governance
- Animal Welfare
- Alternative Proteins
- Global Health
- Pandemic Preparedness & Biosecurity
- Global Poverty & Development
- Existential Risk & Longtermism
- Climate & Environment
- Nuclear Risk
- Space Governance
- Policy & Governance
- Research & Academia
- EA Community & Meta
- Operations & Management

## Architecture

```
sm/
├── __init__.py              # Package exports
├── config.py                # Configuration and EA cause categories
├── cache.py                 # Unified caching system
├── data.py                  # Data loading utilities
├── pipeline.py              # High-level analysis pipeline
├── extractors/
│   ├── base.py              # Base extractor class
│   ├── nlp.py               # spaCy-based extraction
│   ├── llm.py               # Ollama-based extraction with voting
│   └── comparison.py        # Method comparison utilities
├── analysis/
│   ├── geographic.py        # Geocoding and country aggregation
│   ├── semantic.py          # Embedding and clustering
│   └── recommender.py       # Person recommendation system
└── viz/
    ├── maps.py              # Geographic visualizations
    ├── charts.py            # Bar charts, comparisons
    └── network.py           # Semantic network visualization
```

## Configuration

Create a `config.yaml` file in the project root for API credentials:

```yaml
geonames:
  username: your_username

google_maps:
  api_key: your_api_key  # Optional, for enhanced geocoding
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=sm

# Format code
black sm/
ruff check sm/ --fix
```

## License

MIT



# Project notes


## Meta

### Time commitment

The codebase was initially written by hand (~8 hours on and off), after which LLMs were utilised more extensively (see below). This brought the running total to ~20 hours of coding.

### Use of LLMs

There was significant use of LLMs in producing this codebase via the [Cursor](https://cursor.com/agents) IDE. This was justified given the ambition/complexity of the project and needing to balance this with my actual life. It allowed me to explore local implementation of an LLM (new to me); advance my NLP skills (not new to me); build on my software development experience in handling a large project; and practise prompting! As always with LLMs (but for how long I wonder?), the LLM was able to provide 90% of the implementation, after which a good handle on Python code helped tidy things up and prevent unproductive rabbit holes.

### Advancement/Collaboration

I'm signing off on this project for the foreseeable future (the PhD is picking up). However, if anyone is interested in taking it further, please do get in touch with requests/suggestions: and of course free free to fork the repo yourself and have at it: I'd be interested to see where it can be taken!

## Summary 

### General approach

1. **Data Input & Cleaning**
   - Ingest raw attendee data from the .csv file provided via Swapcard 
   - Standardize and clean data fields (strip whitespace, fill NAs, unify date/text formats)
   - Anonymise the data (replace names with unique IDs: the sheet index). This is pretty poor anonymisation but hey this data was given up willingly by participants and isn't exactly compromising. 

2. **Extraction Step**
   - For each row (each corresponding to an attendee), extract three main semantic entities from key text fields:
     - Geographic locations (e.g. countries, cities, regions)
     - Organizations (e.g. universities, companies, NGOs)
     - Cause areas (thematic interests, e.g. "AI safety", "biosecurity")
   - Extraction modules:
     - **NLP Extractor:** spaCy-based, fast, lower recall for "cause area"
     - **LLM Extractor:** Local LLM (via Ollama), higher accuracy, uses majority voting to filter noise

3. **Geocoding**
   - Map extracted geographic locations and organizations to latitude/longitude coordinates
   - Utilize APIs:
     - [GeoNames](https://www.geonames.org/) for countries/regions
     - Google Maps geocoding for organizations/entities
   - Where necessary, apply manual or heuristic filtering to address ambiguous matches

4. **Aggregation**
   - Summarize/analyze field-level results:
     - Count and rank most frequent cause areas, organizations, and locations
     - Compare entity extraction totals and overlaps (NLP vs LLM)
     - Compute Jaccard index for similarity between extractions

5. **Visualization**
   - Generate interactive charts:
     - Bar charts of top cause areas
     - Comparison charts of NLP/LLM results
     - Geographical scatterplots of organizations/locations
   - Visualize overlaps, distributions, and trends

6. **Recommendation (Optional)**
   - Use extracted entities and embeddings to recommend:
     - Similar attendees (by cause area/organization/location)
     - Potential collaborations

7. **Caching & Parallelization**
   - Cache extraction and geocoding results to filesystem to speed up repeated runs
   - Use threads to parallelize LLM requests and geocoding for faster processing

This pipeline ensures each step, from raw data to visualization and recommendation, is modular, testable, and easily extensible for additional data sources or methods.


### Geographic extraction
This is a real pain! The goal was to provide latitude/longitude coordinate pairs for (a) countries mentioned and (b) organisations mentioned in order to visualise the distribution of attendees' geographical connections as well as connecting people based on geography. Turns out there are many organisations with the same name and finding the correct one is non-trivial. At least without using an agentic LLM to browse the web: a whole other kettle of fish...
- The [GeoNames](https://www.geonames.org/) API was used to geocode (provide lat/lon pairs) countries 
- Google provides a (free) [Geocoding API](https://developers.google.com/maps/documentation/geocoding). This was queried and the first search taken as the correct one (since it's ordered by relevance). This gathered the organisation addresses. However, it's not perfect...
- ...due to non-unique organisation names (no, you're not original) there were many false positives. After a few attempts to filter these out (based on whether they were specifically organisations, and whether the result address contains a significant number of words from the organisation name), I lazily added in manual filtering. This is incomplete so there's still a load of silly organisation points.

### General NLP vs LLM
- NLP orders of magnitude faster than LLM, especially due to majority-vote LLM system.
- NLP dramatically underperforms cause area extraction since, for the most part, it only identifies proper nouns. This leads to considerable overestimation (see `extraction_comparison_chart`)
- LLM is a baseline model – there are much bigger and better ones out there. However, since this is a fairly simple semantic classification task created as an MVP, I chose to stay away from SOA models.
- More basic extraction methods (e.g. parsing keywords from their lists) are optionally augmented with the LLM-extracted keywords in the person recommendation system.
- Due to the slight variations in language – which humans would read as basically the same thing – there is poor overlap between the NLP- and LLM-extracted keywords. This is seen by a low [Jaccard metric](https://en.wikipedia.org/wiki/Jaccard_index), where $J$:
  
  $$
  J(A,B) = \frac{A \cap B}{A \cup B}
  $$

  for sets $A$ and $B$, where the index measures similarity between finite non-empty sample sets.

### Future work
- Making this whole thing more accessible via a web-app
- Analysing the databases from past/future conferences to look for (statistically-significant) shifts in interests, expertise, and general focus areas
- Perhaps improving assignment of organisations' locations e.g. via a look-up table. This would be pretty time-consuming to verify and automation would branch into the realms of web-searches...

