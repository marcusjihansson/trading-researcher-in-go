# Enhanced Research Paper Analyzer with DSPy-Go + Advanced Embeddings + AI Model Redundancy

A next-generation research analysis tool that combines PubMed database queries with Google's Gemini AI and Gemma models, enhanced with DSPy-inspired optimization, advanced semantic embeddings, and intelligent model fallback for superior research discovery and analysis.

## 🚀 Key Enhancements

### 🧠 **DSPy-Inspired Optimization**
- **Automatic Query Enhancement**: AI-powered expansion of search terms with domain-specific synonyms
- **Adaptive Prompting**: Self-optimizing prompts that learn from examples
- **Structured Analysis Modules**: Consistent, high-quality research analysis

### 🤖 **AI Model Redundancy**
- **Intelligent Fallback**: Automatic switching from Gemini to Gemma 3 27B during API issues
- **Seamless Continuity**: No interruption when primary model fails or hits rate limits
- **Configurable Backup**: Custom backup models via environment variables
- **Enhanced Reliability**: Always available follow-up question answering

### ⚡ **Advanced Embedding System**
- **Task-Specific Embeddings**: Different embedding types for queries vs documents
- **Multi-Metric Scoring**: Combines semantic similarity, keyword overlap, recency, and journal impact
- **Smart Caching**: Reduces API calls and improves performance
- **Higher Similarity Threshold**: Only shows truly relevant results (>0.35 similarity)

### 🎯 **Enhanced Search Strategy**
- **Query Expansion**: Automatically generates 3-5 related search variations
- **Concept Extraction**: Identifies key scientific concepts and terminology
- **Deduplication**: Combines results from multiple queries intelligently
- **Advanced Ranking**: Multi-factor relevance scoring

## 🆕 New Features

### 📊 **Smart Analysis**
```
## 🎯 RELEVANCE TO QUERY
## 📊 METHODOLOGY & APPROACH
## 🔍 KEY FINDINGS
## 💡 INSIGHTS & IMPLICATIONS
## ⚖️ STRENGTHS & LIMITATIONS
## 🔮 FUTURE RESEARCH DIRECTIONS
## 🏆 SIGNIFICANCE RATING
## 📚 RELATED WORK RECOMMENDATIONS
```

### 🔍 **Multiple Search Options**
- Individual article analysis
- Quick summary of top articles
- Key concepts identification
- Enhanced query exploration
- **Enhanced follow-up question answering** - Ask specific questions about analyzed articles with intelligent fallback to general explanations

### ⚡ **Performance Improvements**
- Embedding caching system
- Rate limiting with smart delays
- Progress indicators
- Performance timing metrics

## 🏗️ Architecture

This project follows a clean, layered architecture with proper separation of concerns:

### **Project Structure**
```
trading_research/
├── cmd/trading-research/
│   └── main.go                    # Application entry point
├── internal/
│   ├── clients/
│   │   └── clients.go            # API client functions
│   ├── config/
│   │   └── config.go             # Configuration constants
│   ├── models/
│   │   └── models.go             # Data structures and types
│   ├── services/
│   │   └── analyzer.go           # Business logic and core functionality
│   └── utils/
│       └── utils.go              # Utility functions
├── go.mod
└── README.md
```

### **Layered Approach**

1. **Query Enhancement Layer**
   - Expands the user's query into multiple semantically related variations
   - Implemented in `QueryEnhancementModule` within services

2. **Retrieval Layer (Concurrent Sources)**
   - Each source is invoked concurrently using goroutines and a WaitGroup
   - A context with timeout bounds the total wait, ensuring the system stays responsive
   - Implemented in `searchPapers` with a fan-out/fan-in pattern

3. **Fusion & Deduplication Layer**
   - Results from all sources are merged and deduplicated by UID (e.g., DOI, arXiv id)

4. **Scoring & Ranking Layer**
   - Embedding-based similarity (cosine) against the original query
   - Optional heuristic bonuses (keyword overlap, recency, journal impact)

5. **Presentation & Analysis Layer**
   - An interactive terminal flow presents top-ranked items
   - Selecting an item performs an LLM-guided structured analysis
   - After analysis, a prompt offers to print a best-effort PDF/DOI/URL link

## 🔧 Installation & Setup

### Prerequisites
- **Go 1.21+** installed
- **Gemini API Key** from [Google AI Studio](https://ai.google.dev/)

### Quick Start
```bash
# 1. Clone or navigate to the project directory
cd trading_research

# 2. Set your API key
export GEMINI_API_KEY="your-gemini-api-key-here"

# 3. Build the application
go build ./cmd/trading-research

# 4. Run the analyzer
./trading-research
```

### Alternative: Run directly
```bash
# Set API key and run in one command
GEMINI_API_KEY="your-key-here" go run ./cmd/trading-research
```

## 📖 Usage Examples

### 🔬 **Basic Research Query**
```
🔍 Enter your research query: machine learning cancer detection

🔍 Enhancing query: machine learning cancer detection
✅ Enhanced query with 4 variations and 6 key concepts
🔍 Searching PubMed for: machine learning cancer detection
🔍 Searching variation 1: deep learning oncology diagnosis
🔍 Searching variation 2: AI-based cancer screening
✅ Found 45 unique articles across all queries

🧠 Generating advanced embeddings and calculating relevance scores...
✅ Found 8 highly relevant articles (similarity > 0.35)
```

### 📚 **Enhanced Article Display**
```
📚 Top Relevant Articles for: "machine learning cancer detection"
💡 Enhanced with: deep learning oncology diagnosis, AI-based cancer screening

[1] Deep Learning for Early Detection of Lung Cancer in CT Scans
    👥 Authors: Zhang, L., Kumar, S., et al.
    📖 Journal: Nature Medicine (2024)
    🎯 Overall Score: 0.912 | Semantic: 0.887 | Keywords: 0.850
    🔗 DOI: 10.1038/s41591-024-12345
    📄 Abstract: We developed a deep learning model that achieves 94.2% accuracy...
```

### 🧠 **Smart Concepts Identification**
```
🧠 KEY CONCEPTS IDENTIFIED
=======================================
Original Query: machine learning cancer detection
Enhanced Queries: deep learning oncology diagnosis | AI-based cancer screening | neural networks tumor identification
Related Concepts: convolutional neural networks | medical imaging AI | predictive oncology | computer-aided diagnosis
```

### ❓ **Enhanced Follow-up Question Answering**
```
🔄 Next action? ('another' for different article, 'ask' for follow-up questions, 'search' for new query, Enter to continue): ask

❓ What question do you have about this article? What is the Put-Call parity mentioned in the paper?

🤔 Answering follow-up question using DSPy-powered analysis...

💡 FOLLOW-UP ANSWER
============================================================
The Put-Call parity is a fundamental principle in options pricing that states the relationship between the prices of European call and put options with the same strike price and expiration date. According to the paper, this parity can be expressed as:

C - P = S - PV(K)

Where:
- C is the call option price
- P is the put option price
- S is the current stock price
- PV(K) is the present value of the strike price K

This relationship holds under the assumptions of no arbitrage opportunities and the ability to create synthetic positions.
============================================================

⚡ Answer generated in 2.34 seconds

🔄 Next action? ('another' for different article, 'ask' for follow-up questions, 'search' for new query, Enter to continue): ask

❓ What question do you have about this article? Please provide me an example of a dynamic valuation model in stock options?

🤔 Answering follow-up question using DSPy-powered analysis...

💡 FOLLOW-UP ANSWER
============================================================
**General Explanation** (Article doesn't cover this specific topic):

A dynamic valuation model for stock options typically refers to the Black-Scholes-Merton model, which revolutionized options pricing. This mathematical model calculates the theoretical value of European-style options using several key inputs:

**Core Formula**: C = S₀N(d₁) - Ke⁻ʳᵀN(d₂)

Where:
- C = Call option price
- S₀ = Current stock price
- K = Strike price
- r = Risk-free interest rate
- T = Time to expiration
- N() = Cumulative normal distribution function
- d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
- d₂ = d₁ - σ√T
- σ = Stock price volatility

**Key Assumptions**: The model assumes lognormal stock price distribution, constant volatility, no transaction costs, and continuous trading. While groundbreaking, it has limitations in real markets where these assumptions don't always hold.

**Practical Applications**: Used by traders for pricing, risk management (delta hedging), and identifying mispriced options in efficient markets.
============================================================

⚡ Answer generated in 1.87 seconds
```

## ⚙️ Configuration

### Environment Variables
- `GEMINI_API_KEY` (required): Your Google Gemini API key
- `BACKUP_MODEL` (optional): Backup AI model for follow-up questions (default: models/gemma-3-27b-it)
- `CORE_API` (optional): Token for CORE v3 API access
- `S2_API_KEY` (optional): Semantic Scholar API key for higher rate limits
- `VERBOSE=1` or `DEBUG=1`: Enable additional debug logging
- `OLLAMA_BASE_URL`: Custom Ollama server URL (default: http://localhost:11434)

### Tuning Parameters
Located in `internal/config/config.go`:
```go
const (
    GeminiAPIBase       = "https://generativelanguage.googleapis.com/v1beta"
    PubMedAPIBase       = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    EmbeddingModel      = "embeddinggemma:300m"
    GenerationModel     = "models/gemini-2.5-flash"
    DefaultBackupModel  = "models/gemma-3-27b-it" // Backup model for follow-up questions
    MaxResults          = 50   // Articles per source
    TopArticles         = 8    // Articles to display
    SimilarityThreshold = 0.35 // Minimum relevance threshold
)
```

## 🔍 Source Integrations

The system queries multiple academic sources concurrently:
- **arXiv**: Quantitative finance papers (q-fin categories)
- **Crossref**: Journal articles with DOI metadata
- **Semantic Scholar**: Academic papers with rich metadata
- **CORE**: Open access research papers

All sources are queried in parallel with a 6-second timeout for optimal performance.

### PDF/DOI/URL Link Resolution
After analysis, the system provides the best available link with this priority:
1. arXiv UID → `https://arxiv.org/pdf/<id>.pdf`
2. Bare DOI (`10.x/...`) → `https://doi.org/<DOI>`
3. Existing URLs (with arXiv abs→pdf conversion)

## 🚀 Advanced Features

### 🤖 **Intelligent Model Fallback**
The follow-up question system includes automatic model redundancy:
- **Primary Model**: Gemini 2.5 Flash for optimal performance
- **Backup Model**: Gemma 3 27B automatically activates if Gemini fails
- **Seamless Switching**: No user intervention required during API issues or rate limits
- **Configurable Backup**: Set `BACKUP_MODEL` environment variable for custom backup models

### DSPy Module Optimization
The system learns from examples to improve query enhancement:

```go
// Add domain-specific optimization examples
analyzer.optimizationData = []models.Example{
    {
        Inputs: map[string]interface{}{
            "original_query": "your research area",
        },
        Outputs: map[string]interface{}{
            "response": `{"expanded_queries": [...], "key_concepts": [...]}`,
        },
    },
}
```

### Embedding Caching
- Reuses embeddings for identical text to reduce API calls
- Task-specific embeddings (query vs document)
- Automatic cache management

## 🐛 Troubleshooting

### Common Issues & Solutions

**🚫 "No highly relevant articles found"**
```
Solution: The system uses similarity thresholds (>0.35)
- Try broader search terms
- Use more general concepts
- Check if your query is too specific
```

**⚠️ "Query enhancement failed"**
```
Solution: System gracefully falls back to original query
- Check your API key quota
- Verify internet connection
- System continues with basic search
```

**🔄 "Rate limit exceeded"**
```
Solution: Built-in rate limiting and model fallback prevent most issues
- System automatically switches to backup model (Gemma 3 27B) if Gemini is rate limited
- Wait 60 seconds and retry for full service restoration
- Check your Gemini API quota
- Consider upgrading API plan for heavy usage
```

### Performance Tuning

**Speed up searches:**
- Reduce `MaxResults` for faster processing
- Lower `TopArticles` for quicker selection
- Increase `SimilarityThreshold` for fewer but better results

**Improve accuracy:**
- Add domain-specific optimization examples
- Customize journal impact scoring
- Adjust similarity calculation weights

## 📊 Example Session Output

```
🔬 Enhanced Research Paper Analyzer with DSPy-Go + Advanced Embeddings
=======================================================================
🚀 Features: Query Enhancement | Advanced Embeddings | Smart Ranking
🧠 Powered by: Gemini Flash + Embeddings + DSPy-Inspired Optimization

✅ System initialized with DSPy optimization examples

===========================================================

🔍 Enter your research query: CRISPR gene therapy

🔍 Enhancing query: CRISPR gene therapy
✅ Enhanced query with 3 variations and 5 key concepts
🔍 Searching PubMed for: CRISPR gene therapy
✅ Found 18 articles
🔍 Searching variation 1: CRISPR-Cas9 therapeutic applications
✅ Found 22 articles
🔍 Searching variation 2: gene editing clinical trials
✅ Found 31 unique articles across all queries

🧠 Generating advanced embeddings and calculating relevance scores...
Processing article 31/31...
✅ Found 6 highly relevant articles (similarity > 0.35)

📚 Top Relevant Articles for: "CRISPR gene therapy"
💡 Enhanced with: CRISPR-Cas9 therapeutic applications, gene editing clinical trials
=================================================================================

[1] CRISPR-Cas9 Gene Editing for Sickle Cell Disease: Clinical Trial Results
    👥 Authors: Frangoul, H., Altshuler, D., et al.
    📖 Journal: New England Journal of Medicine (2024)
    🎯 Overall Score: 0.934 | Semantic: 0.901 | Keywords: 0.920
    🔗 DOI: 10.1056/NEJMoa2031713
    📄 Abstract: We report the results of a phase 3 clinical trial evaluating CRISPR-Cas9...

⚡ Analysis completed in 12.34 seconds
📊 Processed 31 total articles, found 6 highly relevant matches

🎯 Selection Options:
   1-6: Analyze specific article
   'summary': Get quick summary of all top articles
   'concepts': Show key concepts found
   'new': Start new search

📖 Your choice: 1

✅ Selected: CRISPR-Cas9 Gene Editing for Sickle Cell Disease: Clinical Trial Results
🎯 Relevance Score: 0.934

📊 Analyzing article with enhanced DSPy-powered analysis...

📊 COMPREHENSIVE ANALYSIS
=========================================================================================
## 🎯 RELEVANCE TO QUERY
This article is highly relevant to "CRISPR gene therapy" as it presents clinical trial results
for CRISPR-Cas9 gene editing as a therapeutic intervention for sickle cell disease...

## 📊 METHODOLOGY & APPROACH
This phase 3, multicenter, randomized controlled trial enrolled 75 patients with severe sickle
cell disease. The study utilized CRISPR-Cas9 technology to edit the BCL11A gene...

[Full detailed analysis continues...]
=========================================================================================

⚡ Analysis generated in 4.21 seconds
```

## 🔄 Data Flow Architecture

```
User Query → Query Enhancement Module
    ↓
Enhanced Queries → Multiple Concurrent Searches
    ↓
Combined Articles → Advanced Embedding Analysis
    ↓
Ranked Results → User Selection
    ↓
Selected Article → DSPy Analysis Module
    ↓
Structured Analysis → User Display
```

## 🤝 Contributing & Customization

This enhanced system is designed to be extensible:

- **Add new DSPy modules** for specialized analysis types
- **Customize scoring algorithms** for specific research domains
- **Extend embedding strategies** for different document types
- **Implement learning mechanisms** to improve over time

## 📄 License

Socials: 
LinkedIn: https://www.linkedin.com/in/marcus-frid-johansson/
X/Twitter: https://x.com/marcusjihansson
