# LLM Engineering Course - Week 4: Choosing the Right LLM & Code Generation

## Course Overview
This is Week 4 of an 8-week LLM Engineering course focusing on how to select the right Large Language Model for specific tasks and practical code generation applications.

**8-Week Course Structure:**
- Week 1: Frontier Models comparison
- Week 2: UIs with Gradio, agents, multi-modality
- Week 3: Open source with Hugging Face
- **Week 4: Selecting LLMs & Code Generation** (Current)
- Week 5: RAG (Retrieval Augmented Generation)
- Week 6: Fine-tuning
- Week 7-8: Training (frontier and open source)

---

## Key Principle: No "Best" LLM
**Main Takeaway:** There's no single "best" LLM. The right choice depends on your specific task and requirements.

---

## Part 1: Comparing LLMs - Basic Attributes

### Open Source vs Closed Source Decision
**First decision point:** Choose between open and closed source models.

**Open Source Models:**
- Code is publicly available
- Can be modified and customized
- Often free to use (with some license restrictions)
- Examples: Llama, Mistral, Qwen

**Closed Source Models:**
- Proprietary code, accessed via APIs
- Generally more powerful but costly
- Examples: GPT-4, Claude, Gemini

### Essential Basic Attributes to Compare

1. **Release Date & Knowledge Cutoff**
   - When was the model released?
   - What's the last date of its training data?
   - Important for current events awareness

2. **Model Size (Parameters)**
   - Number of parameters indicates model strength
   - More parameters = more powerful but more expensive
   - Affects training costs for fine-tuning

3. **Training Data Size**
   - Number of tokens used during training
   - Indicates depth of knowledge

4. **Context Length (Context Window)**
   - Maximum tokens the model can "remember" at once
   - Includes system prompts, user inputs, and conversation history
   - Example: Gemini 1.5 Flash has 1 million tokens (longest available)

### Cost Considerations

**Inference Costs** (running the model):
- API costs for closed source models
- Subscription fees for chat interfaces
- Compute costs for self-hosted open source models

**Training Costs:**
- Zero for out-of-the-box models
- Significant for fine-tuning custom models

**Build Costs:**
- Time and effort to implement
- Closed source = faster time to market
- Open source = more customization but longer setup

### Performance Factors

**Speed & Latency:**
- Speed: How fast tokens are generated
- Latency: Time until first response
- Critical for real-time applications

**Rate Limits & Reliability:**
- API usage restrictions
- Service availability issues during high demand

**Licensing:**
- Usage restrictions and commercial limitations
- Some models require business agreements above certain revenue levels

---

## Part 2: Advanced Evaluation - Benchmarks & Leaderboards

### The Chinchilla Scaling Law
**Simple Explanation:** A rule discovered by Google DeepMind that shows the relationship between model size and training data.

**Key Insight:** If you double the training data, you need to double the model parameters to effectively use that extra data.

**Practical Use:**
- Helps estimate how much training data you need for a given model size
- Guides decisions about scaling up models

### Common Benchmarks (Tests for LLMs)

**Think of benchmarks as standardized tests that measure different abilities:**

1. **ARC** - Scientific reasoning (multiple choice science questions)
2. **DROP** - Reading comprehension with math (read text, then count/sort/add from it)
3. **HellaSwag** - Common sense reasoning
4. **MMLU** - General knowledge across 57 subjects (like a comprehensive exam)
5. **TruthfulQA** - Accuracy when pressured to be untruthful
6. **Winograd** - Resolving confusing language contexts
7. **GSM8K** - Grade school math word problems

### Specialized Benchmarks

**For Chat Evaluation:**
- **ELO Rating** - Like chess rankings, based on head-to-head comparisons by humans

**For Coding:**
- **HumanEval** - 164 Python programming problems
- **MultiPL-E** - Coding tests in 18 different programming languages

### Next-Level (Harder) Benchmarks

1. **GPQA (Google-Proof Q&A)**
   - PhD-level questions in physics, chemistry, biology
   - So hard that even with Google, most people score only 34%
   - PhD experts average 65%
   - Claude 3.5 Sonnet currently leads at 59.4%

2. **BBHard (Big Bench Hard)**
   - Originally designed to test "future" AI capabilities
   - Models have already mastered it (showing rapid AI progress)

3. **MATH Level 5**
   - High school math competition problems (extremely difficult)

4. **IFEval**
   - Tests ability to follow complex instructions
   - Example: "Write more than 400 words and mention 'I' at least 3 times"

5. **MUSR (Multi-step Soft Reasoning)**
   - Analyzes 1000-word murder mysteries
   - AI must determine who has means, motive, and opportunity

6. **MMLU-Pro**
   - Improved version of MMLU with harder questions
   - 10 answer choices instead of 4 (reduces lucky guessing)

### Limitations of Benchmarks

**Important Warnings:**

1. **Inconsistent Application**
   - Different organizations may test differently
   - No standardized testing conditions

2. **Training Data Leakage**
   - Models might have seen the test questions during training
   - Makes scores artificially high

3. **Overfitting**
   - Models tuned specifically to perform well on benchmarks
   - May not perform well on similar but different tasks

4. **Too Narrow**
   - Multiple choice questions don't test nuanced reasoning
   - Real-world tasks are more complex

5. **Evaluation Awareness** (New concern)
   - Advanced models might know they're being tested
   - Could affect responses, especially on safety/alignment tests

---

## Part 3: Essential Leaderboards & Resources

### Must-Bookmark Resources

1. **Hugging Face OpenLLM Leaderboard**
   - Primary resource for comparing open source models
   - Uses the harder, next-level benchmarks
   - Filter by model type, parameter size, precision
   - **Key Models Leading:** Qwen2, Llama 3.1, Phi-3, Mistral

2. **Specialized Hugging Face Leaderboards:**
   - **Big Code Models** - For programming tasks
   - **LMPerf** - Performance vs. accuracy trade-offs
   - **Medical Leaderboard** - Healthcare-specific benchmarks
   - **Language-specific** - Portuguese, Spanish, etc.

3. **Vellum AI Leaderboard**
   - Combines open and closed source models
   - Shows cost comparisons and context window sizes
   - Performance metrics (speed, latency, cost per million tokens)

4. **Scale AI SEAL Leaderboards**
   - Business-specific benchmarks
   - Adversarial robustness, instruction following
   - Regular additions of new specialized tests

5. **LMSys Chatbot Arena**
   - Human-evaluated chat performance
   - Blind voting system (you don't know which model is which)
   - ELO ratings based on head-to-head comparisons
   - **Current Leader:** ChatGPT-4o

---

## Part 4: Real-World Commercial Applications

### Example Use Cases

1. **Legal (Harvey AI)**
   - Answering legal questions
   - Document analysis and key term extraction

2. **Recruitment (Nebula.io)**
   - Matching candidates to roles
   - Career guidance and satisfaction prediction

3. **Legacy Code Conversion (Bloop AI)**
   - Converting COBOL to modern languages like Java
   - Addressing shortage of legacy language programmers

4. **Healthcare (Salesforce Einstein)**
   - Medical appointment summaries
   - Care coordination assistance

5. **Education (Khan Academy's Conmigo)**
   - AI companion for teachers, students, and parents
   - Personalized learning assistance

---

## Part 5: Practical Code Generation Project

### The Challenge: Python to C++ Conversion

**Goal:** Build a tool that converts Python code to optimized C++ for better performance.

### Simple Test Case: Calculating Pi
**Python Code:** Calculate pi using the series: 1 - 1/3 + 1/5 - 1/7 + 1/9...

**Results:**
- Python: 8.57 seconds
- GPT-4 generated C++: 0.21 seconds (~40x faster)
- Claude 3.5 Sonnet C++: 0.21 seconds (~40x faster)

**Key Learning:** Both frontier models produced similarly optimized code for simple problems.

### Complex Test Case: Maximum Subarray Problem
**Challenge:** Find the largest sum of consecutive numbers in an array of positive and negative integers.

**Python Implementation:** Nested loops trying all possible subarrays (inefficient)

**Results:**
- Python: 27 seconds
- GPT-4: Failed (number overflow errors, returned 0)
- Claude 3.5 Sonnet: 0.0004 seconds (67,500x faster!)

**Why Claude Won:**
Claude didn't just translate the code - it **understood the algorithm's intent** and reimplemented it using Kadane's Algorithm (single loop instead of nested loops).

### Key Insights from Code Generation

1. **Claude 3.5 Sonnet excels at coding tasks**
   - Consistently ranks #1 in coding leaderboards
   - Better algorithmic understanding
   - More reliable code generation

2. **GPT-4 needed more specific prompting**
   - Required hints about number types and includes
   - More prone to implementation errors

3. **Frontier models can optimize, not just translate**
   - Best models understand problem intent
   - Can suggest better algorithms, not just convert syntax

---

## Part 6: Building the Complete Solution

### Gradio UI Development
**Components Built:**
- Python code input
- Model selection (GPT-4 vs Claude)
- C++ code generation
- Side-by-side execution and timing
- Performance comparison display

**Safety Note:** The exec() function used is dangerous in production - only suitable for personal prototypes.

### Performance Results Summary

| Task | Python Time | GPT-4 C++ | Claude C++ | Speed Improvement |
|------|-------------|-----------|------------|-------------------|
| Simple Pi Calculation | 8.57s | 0.21s | 0.21s | ~40x |
| Complex Algorithm | 27s | Failed | 0.0004s | 67,500x |

---

## Part 7: Open Source Models for Code Generation

### Hugging Face Inference Endpoints

**What are Inference Endpoints?**
Think of them as "rental computers" in the cloud that run your chosen AI model. Instead of downloading and running a huge model on your own computer, Hugging Face runs it for you and gives you a web address (endpoint) to send requests to.

**Key Benefits:**
- No need to download multi-gigabyte models
- Professional-grade hardware (GPUs) provided
- Pay only for what you use
- Easy to set up and tear down

### Top Open Source Code Generation Models

**Based on Big Code Models Leaderboard:**

1. **CodeQwen 1.5 7B Chat** (Top performer)
   - 7 billion parameters
   - Designed for conversational code generation
   - Excellent at Python and C++ tasks
   - Can follow chat-style instructions

2. **Code Llama variants**
   - Different sizes available (7B, 13B, 34B)
   - Meta's specialized coding model
   - Good general-purpose coding abilities

3. **DeepSeek Coder**
   - Strong performance on coding benchmarks
   - Good at understanding coding context

### Setting Up Inference Endpoints

**Step-by-Step Process:**
1. Go to the model page on Hugging Face
2. Click "Deploy" → "Inference Endpoints"
3. Choose cloud provider (AWS/Azure/GCP)
4. Select hardware (GPU recommended for code models)
5. Click "Create Endpoint"
6. Wait ~5 minutes for deployment

**Cost Example:**
- NVIDIA L4 GPU: ~$0.80/hour
- Can be paused when not in use
- Good for experiments and testing

### Technical Implementation

**Key Concepts Explained:**

1. **Tokenizer**: 
   - Converts text into numbers the model understands
   - Each model has its own tokenizer
   - Like a translator between human language and AI language

2. **Chat Templates**:
   - Special formatting that tells the model how to interpret conversations
   - Wraps user messages and system instructions properly
   - Different models use different templates

3. **Inference Client**:
   - Python code that sends requests to your endpoint
   - Handles the communication between your code and the remote model
   - Returns streaming or complete responses

**Code Structure:**
```python
# Create tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")

# Format conversation properly
text = tokenizer.apply_chat_template(messages, tokenize=False)

# Send to endpoint
client = InferenceClient(url="your-endpoint-url", token="your-token")
response = client.text_generation(text, stream=True, max_new_tokens=1000)
```

---

## Part 8: Open Source vs Frontier Model Results

### CodeQwen Performance Analysis

**Simple Pi Calculation Test:**
- ✅ Successfully converted Python to C++
- ✅ Generated optimized, working code
- ✅ Similar performance to GPT-4 (~40x speedup)
- ⚠️ Added explanatory text despite instructions not to

**Complex Algorithm Test (Maximum Subarray):**
- ❌ Changed random number generation approach
- ❌ Results didn't match original Python output
- ❌ Couldn't validate correctness due to number mismatch
- ⚠️ Showed algorithmic understanding but failed consistency requirements

### Key Findings

**Where Open Source Succeeded:**
- Simple code translation tasks
- Basic optimization
- Clean, readable C++ output
- Cost-effective solution ($0.80/hour vs API costs)

**Where Open Source Struggled:**
- Following specific constraints (random number generation)
- Complex algorithmic reimplementation
- Instruction adherence (adding unwanted explanations)
- Maintaining exact numerical consistency

**Parameter Count Context:**
- CodeQwen: 7 billion parameters
- GPT-4: >1 trillion parameters (estimated)
- Claude 3.5: >1 trillion parameters (estimated)

**Fair Assessment:**
Given the massive parameter difference, CodeQwen performed remarkably well. For many practical code generation tasks, it would be sufficient and cost-effective.

---

## Part 9: Performance Evaluation Metrics

### Two Types of Metrics

Understanding how to measure AI success is crucial for any project. There are two main categories:

### 1. Model-Centric (Technical) Metrics

**What they are:** Direct measurements of how well the AI model performs on its core task.

**Key Technical Metrics:**

1. **Cross-Entropy Loss**
   - **Simple explanation:** Measures how surprised the model is by the correct answer
   - **How it works:** When predicting the next word, the model gives probabilities for all possible words. Loss looks at how much probability it gave to the actual correct word.
   - **Scale:** Lower is better (0 = perfect prediction)
   - **Use:** Training optimization and model comparison

2. **Perplexity**
   - **Simple explanation:** How "confused" the model is when making predictions
   - **Formula:** e^(cross-entropy loss)
   - **Interpretation:**
     - Perplexity of 1 = Perfect confidence and accuracy
     - Perplexity of 2 = Like flipping a coin (50/50 chance)
     - Perplexity of 4 = 25% accuracy if guessing randomly
   - **Use:** Comparing language model quality

**Why Technical Metrics Matter:**
- Fast to calculate during training
- Help optimize model parameters
- Allow comparison between different models
- Guide technical decisions

### 2. Business-Centric (Outcome) Metrics

**What they are:** Measurements of real-world impact and business value.

**Examples from Our Code Generation Project:**
- **Speed Improvement:** How much faster is the generated C++ vs original Python?
- **Accuracy:** Does the output produce the same results as the input?
- **Success Rate:** What percentage of code conversions work correctly?

**Other Business Metric Examples:**
- **Customer Satisfaction:** User ratings of AI chatbot interactions
- **Cost Reduction:** Money saved through automation
- **Time Savings:** Hours saved per week using the AI tool
- **Revenue Impact:** Additional sales generated by AI recommendations
- **Error Reduction:** Decrease in human mistakes when using AI assistance

**Why Business Metrics Matter:**
- Show actual value delivered
- Justify investment in AI projects
- Resonate with stakeholders and executives
- Measure real-world impact

### The Balanced Approach

**Use Both Types Together:**
- **Technical metrics** help you build better models
- **Business metrics** prove the solution works in practice
- Technical improvements should lead to business improvements
- Business needs should guide technical optimization

**Example from Our Project:**
- **Technical:** Model generates syntactically correct C++
- **Business:** C++ runs 40x faster than original Python
- **Combined insight:** Technical correctness enables business value

---

## Part 10: Advanced Challenges and Projects

### Challenge 1: Automatic Code Documentation

**Project Goal:** Build a tool that automatically adds comments and docstrings to existing code.

**Why This Matters:**
- Poor documentation is a major problem in software development
- Saves developers hours of manual work
- Improves code maintainability
- Makes codebases more accessible to new team members

**Technical Approach:**
1. Parse existing Python/JavaScript/etc. files
2. Send functions to LLM with instructions to add documentation
3. Format results according to language standards (PEP 257 for Python)
4. Preserve original code structure and functionality

**Success Metrics:**
- **Technical:** Generated docstrings follow proper format standards
- **Business:** Time saved in code review process, improved code readability scores

### Challenge 2: Automated Unit Test Generation

**Project Goal:** Generate comprehensive unit tests for Python modules.

**Why This Matters:**
- Unit testing is often neglected due to time pressure
- Good tests catch bugs early and save debugging time
- Test coverage improves code quality and reliability
- Automated generation ensures consistency

**Technical Approach:**
1. Analyze function signatures and behavior
2. Generate test cases covering edge cases, normal inputs, and error conditions
3. Use proper testing frameworks (pytest, unittest)
4. Include setup and teardown code where needed

**Success Metrics:**
- **Technical:** Test coverage percentage, test case diversity
- **Business:** Bugs caught before production, reduced debugging time

### Challenge 3: AI Trading Signal Generator

**Project Goal:** Build a system that generates trading algorithms (for simulation only).

**⚠️ Important Warning:** This is for educational and simulation purposes only. Never use AI-generated trading strategies with real money. Financial markets are complex and risky.

**Why This Is Interesting:**
- Combines code generation with domain-specific knowledge
- Tests model's ability to understand financial concepts
- Provides measurable performance metrics (simulated returns)
- Demonstrates real-world complexity of AI applications

**Technical Approach:**
1. Define a mock trading API (get_price, place_order, etc.)
2. Provide LLM with API documentation and market context
3. Generate trading functions that make buy/sell decisions
4. Test strategies in simulated environment with historical data

**Example API Functions to Provide:**
```python
def get_current_price(ticker: str) -> float:
    """Returns current stock price"""
    
def get_historical_prices(ticker: str, days: int) -> list:
    """Returns price history"""
    
def place_order(ticker: str, action: str, quantity: int):
    """Places buy/sell order"""
    
def get_portfolio_value() -> float:
    """Returns total portfolio value"""
```

**Success Metrics:**
- **Technical:** Code compiles and runs without errors
- **Business:** Simulated returns vs baseline (buy-and-hold)
- **Risk:** Maximum drawdown, volatility metrics

### Implementation Tips for All Challenges

**Model Selection Strategy:**
1. Start with frontier models (Claude 3.5, GPT-4) for baseline
2. Test open source alternatives for cost comparison
3. Consider using different models for different complexity levels

**Prompt Engineering:**
- Be specific about output format requirements
- Include examples of desired output
- Add constraints about what NOT to do
- Test prompts with simple cases first

**Error Handling:**
- Plan for model failures and malformed outputs
- Include validation steps for generated code
- Have fallback strategies for edge cases

**Evaluation Framework:**
- Define success criteria upfront
- Create test cases of varying difficulty
- Track both technical and business metrics
- Compare multiple model approaches

---

## Key Takeaways for LLM Selection

### 1. No Universal Best Model
Choose based on specific requirements:
- **Task complexity:** Simple tasks may work fine with smaller models
- **Cost constraints:** Open source for budget-conscious projects
- **Performance needs:** Frontier models for critical applications
- **Customization requirements:** Open source for specialized fine-tuning

### 2. Use Multiple Evaluation Methods
- **Benchmarks:** Good for general capability comparison
- **Leaderboards:** Show relative performance rankings
- **Real testing:** Always test with your actual use case
- **Business metrics:** Measure actual impact, not just technical performance

### 3. Consider Total Cost of Ownership
Not just API costs, but also:
- Development time and complexity
- Infrastructure and hosting costs
- Maintenance and updates
- Performance optimization needs

### 4. Test with Your Actual Use Case
- Benchmarks don't tell the whole story
- Your specific requirements may favor different models
- Edge cases in your domain may not be covered by general benchmarks
- User experience factors may override pure performance metrics

### 5. Model-Specific Strengths
- **Claude 3.5 Sonnet:** Excels at code generation and complex reasoning
- **GPT-4:** Strong general capabilities, good for diverse tasks
- **Open Source (CodeQwen, Llama):** Cost-effective, customizable, privacy-friendly
- **Specialized models:** Consider domain-specific fine-tuned models

### 6. Context Windows Matter
- Ensure sufficient space for your use case
- Consider conversation history, system prompts, and output space
- Longer contexts enable more sophisticated applications
- Balance context length with cost and performance

### 7. Benchmark Limitations
Take scores with healthy skepticism:
- Training data contamination possible
- Narrow evaluation scope
- Gaming of specific benchmarks
- Real-world performance may vary significantly

### 8. Open Source Is Competitive
- Models like Llama 3.1 405B rival closed source capabilities
- Significant cost advantages for high-volume applications
- Full control over deployment and customization
- Rapid improvement and community development

---

## Course Progress Milestone

🎉 **Congratulations!** You've reached the 50% milestone in your journey to LLM Engineering mastery.

**Skills Acquired So Far:**
- ✅ Understanding of frontier vs open source models
- ✅ Benchmark and leaderboard interpretation
- ✅ Practical code generation implementation
- ✅ Hugging Face Inference Endpoints deployment
- ✅ Performance evaluation methodologies
- ✅ Multi-model comparison frameworks
- ✅ Real-world application assessment

**What's Next:**
- **Week 5:** RAG (Retrieval Augmented Generation) - Adding external knowledge to LLMs
- **Week 6:** Fine-tuning - Customizing models for specific tasks
- **Week 7-8:** Training - Building models from scratch

**Your Learning Path Forward:**
1. Complete the challenge projects to solidify understanding
2. Experiment with different models and use cases
3. Build a portfolio of LLM applications
4. Share your results and learn from others

The journey continues with even more exciting topics ahead!
