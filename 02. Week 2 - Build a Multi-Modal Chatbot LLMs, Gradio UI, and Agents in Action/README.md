# Week 2: LLM Engineering Program - Complete Learning Guide

## Program Overview
This is Week 2 of an 8-week LLM (Large Language Model) Engineering program. By the end of this week, students will be 25% through their journey to becoming LLM Engineering Masters.

**Week Structure:**
- **Week 1**: Foundation (Transformers, APIs, basic concepts) ✅
- **Week 2**: Multiple APIs + UI Development 📍 Current
- **Week 3**: Agent-ization and multi-modality
- **Week 4**: Open source with Hugging Face
- **Week 5**: Selecting the right LLM and code generation
- **Week 6**: RAG (Retrieval Augmented Generation)
- **Week 7**: Fine-tuning (frontier and open source)
- **Week 8**: Finale

---

## Day 1: Mastering Multiple AI APIs

### **Core Concept: API Integration**
**What it is:** Learning to work with different AI service providers through their programming interfaces.

**Key Skills Developed:**
- Using OpenAI API (GPT models)
- Using Anthropic API (Claude models) 
- Using Google API (Gemini models)
- Understanding streaming responses
- Building conversational AI systems

### **Important Concepts Explained:**

#### **1. Streaming Responses**
**Simple Explanation:** Instead of waiting for the entire AI response to be generated, you receive it piece by piece in real-time, like watching someone type.

**Why it matters:** 
- Better user experience (no long waits)
- Feels more natural and conversational
- Users see progress happening

**Technical Implementation:**
```python
# OpenAI streaming
response = openai.chat.completions.create(
    model="gpt-4-mini",
    messages=messages,
    stream=True  # This enables streaming
)

# Claude streaming  
response = claude.messages.stream(  # Different method for Claude
    model="claude-3-haiku",
    messages=messages
)
```

#### **2. Message Structure**
**Simple Explanation:** A standardized way to organize conversations between humans and AI, like keeping a chat history.

**Structure:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Tell me a joke"}
]
```

**Roles Explained:**
- **System**: Sets the AI's personality and instructions
- **User**: What the human says
- **Assistant**: What the AI responds

#### **3. Temperature Setting**
**Simple Explanation:** Controls how creative vs predictable the AI's responses are.
- **0**: Very focused and consistent (like a careful student)
- **1**: More creative and random (like a creative artist)

---

## Day 2: Building AI UIs with Gradio

### **Core Concept: Rapid UI Development**
**What it is:** Creating user interfaces for AI applications without complex web development.

### **Important Concepts Explained:**

#### **1. Gradio Framework**
**Simple Explanation:** A Python library that lets you create web interfaces for AI models with just a few lines of code.

**Why it's revolutionary:**
- Traditional web development requires HTML, CSS, JavaScript, and backend setup
- Gradio turns a simple Python function into a web app instantly
- Perfect for prototyping and sharing AI tools

**Basic Example:**
```python
import gradio as gr

def my_function(text):
    return text.upper()

# This single line creates a web interface!
gr.Interface(fn=my_function, inputs="text", outputs="text").launch()
```

#### **2. Interface Components**
**Simple Explanation:** Ready-made UI elements you can use without coding them from scratch.

**Common Components:**
- **Text boxes**: For user input
- **Markdown**: For formatted output
- **Dropdowns**: For model selection
- **Buttons**: For actions

#### **3. Share Feature**
**Simple Explanation:** When you set `share=True`, Gradio creates a public URL that anyone can access to use your AI tool.

**Amazing Feature:** The code still runs on your local computer, but others can access the interface through the internet.

---

## Day 3: Building AI Chatbots

### **Core Concept: Conversational AI Systems**
**What it is:** Creating AI assistants that can maintain context across multiple exchanges, like a real conversation.

### **Important Concepts Explained:**

#### **1. Chat Interface Structure**
**Simple Explanation:** A special way to organize conversations that Gradio understands for chat-style interfaces.

**Function Requirements:**
```python
def chat(message, history):
    # message: current user input
    # history: list of [user_msg, assistant_msg] pairs
    return response
```

#### **2. Context Maintenance**
**Simple Explanation:** How AI "remembers" what was said earlier in the conversation.

**The Reality:** 
- AI doesn't actually remember anything
- Every time you send a message, the ENTIRE conversation history is sent to the AI
- The AI processes everything from the beginning each time
- This creates the illusion of memory

**Why This Matters:**
- Context windows (memory limits) include all previous messages
- Longer conversations cost more to process
- Eventually, very long conversations hit memory limits

#### **3. Multi-Shot Prompting**
**Simple Explanation:** Giving the AI multiple examples of how to behave, like showing a student several solved problems before asking them to solve a new one.

**Example:**
```python
system_prompt = """You are a store assistant. Here are examples:

Customer: "I want shoes"
You: "Great! Shoes aren't on sale, but our hats are 60% off!"

Customer: "Do you have belts?"
You: "Sorry, no belts, but check out our amazing hat selection!"
"""
```

#### **4. Dynamic Context Addition**
**Simple Explanation:** Adding relevant information to conversations based on what the user mentions.

**Example Use Case:**
- User mentions "belt"
- System automatically adds: "The store doesn't sell belts, but highlight sale items"
- This is like a basic version of RAG (Retrieval Augmented Generation)

---

## Day 4: AI Tools and Function Calling

### **Core Concept: Extending AI Capabilities**
**What it is:** Teaching AI to use external functions and tools, like giving it a calculator or database access.

### **Important Concepts Explained:**

#### **1. How Tools Actually Work**
**Simple Explanation:** It's not magic! Here's the real process:

1. **You define a function** (like a calculator)
2. **You tell the AI about it** with a description
3. **AI responds:** "I need to use the calculator tool with these inputs"
4. **You run the function** and give results back to AI
5. **AI uses the results** to create its final response

**It's basically a fancy if-statement system!**

#### **2. Tool Definition Structure**
**Simple Explanation:** A specific format to describe your functions so the AI understands when and how to use them.

```python
tool_definition = {
    "type": "function",
    "function": {
        "name": "get_ticket_price",
        "description": "Get flight prices for cities. Call when user asks about ticket costs.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination_city": {
                    "type": "string", 
                    "description": "The city to get prices for"
                }
            },
            "required": ["destination_city"]
        }
    }
}
```

#### **3. The Tool Workflow**
**Simple Explanation:** Step-by-step process of how AI uses tools:

1. User asks: "How much is a ticket to London?"
2. AI thinks: "I need price information, I should use the ticket tool"
3. AI responds: `{"tool": "get_ticket_price", "args": {"city": "London"}}`
4. Your code runs: `get_ticket_price("London")` → returns "$799"
5. You send back: `{"role": "tool", "content": "London: $799"}`
6. AI responds: "A ticket to London costs $799"

#### **4. Common Tool Use Cases**
**Simple Explanation:** Popular ways to extend AI capabilities:

- **Data Fetching**: Look up information in databases
- **Actions**: Book meetings, send emails, make purchases
- **Calculations**: Math, data analysis, complex computations
- **UI Changes**: Update interfaces, modify displays

---

## Day 5: Multimodal AI and Agents

### **Core Concept: Advanced AI Systems**
**What it is:** Creating AI that can work with images, sound, and coordinate multiple specialized tasks.

### **Important Concepts Explained:**

#### **1. Multimodal AI**
**Simple Explanation:** AI that can work with different types of content, not just text.

**Types:**
- **Text-to-Image**: Creating pictures from descriptions (DALL-E)
- **Text-to-Audio**: Generating sounds or speech
- **Image-to-Text**: Describing what's in pictures
- **Video Processing**: Understanding or creating videos

**Technical Note:** While we call them "Language Models," image and audio generation models aren't technically language models - they're specialized models for different types of content. However, they're part of the modern LLM engineer's toolkit.

#### **2. DALL-E 3 Image Generation**
**Simple Explanation:** An AI model that creates pictures from text descriptions.

**Key Features:**
- **Cost**: $0.04 per image (higher cost than text generation)
- **Quality**: Very high-quality, creative images
- **Sizes**: Multiple sizes available (smallest to largest)
- **Format**: Returns images in base64 encoded format

**Example Workflow:**
```python
def artist(city):
    response = openai.images.generate(
        model="dalle-3",
        prompt=f"An image representing a vacation in {city}",
        size="1024x1024",
        quality="standard",
        n=1
    )
    # Convert base64 to displayable image
    return image
```

#### **3. Text-to-Speech (TTS)**
**Simple Explanation:** Converting written text into spoken audio.

**Key Features:**
- **Model**: TTS-1 (Text-to-Speech version 1)
- **Voices**: Multiple voice options (Onyx, Alloy, etc.)
- **Output**: Audio that can be played directly

**Example Implementation:**
```python
def talker(text):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )
    # Convert to playable audio
    return audio
```

#### **4. AI Agents**
**Simple Explanation:** Specialized AI programs that can work independently to accomplish specific tasks.

**Five Types of Agent Systems:**
1. **Multi-step Problem Solving**: Breaking complex problems into smaller tasks with different models handling each part
2. **Tool-Enhanced LLMs**: Giving AI access to external functions and capabilities
3. **Agent Environment**: The framework that allows multiple agents to collaborate
4. **Planning Agents**: One LLM acts as a coordinator, dividing work among specialist agents
5. **Autonomous Agents**: AI that can act independently, like monitoring news and making trading decisions

#### **5. Agent Framework Architecture**
**Simple Explanation:** A system where multiple specialized AI components work together.

**Example Multi-Agent Workflow:**
1. **User Input**: "I want to book a ticket to London"
2. **Planning Agent**: Decides what tools/agents are needed
3. **Price Tool**: Looks up ticket costs
4. **Image Agent**: Generates city image using DALL-E
5. **Audio Agent**: Converts response to speech
6. **Response Coordination**: Combines all outputs into final response

#### **6. Multimodal Integration Patterns**
**Simple Explanation:** Common ways to combine different AI capabilities:

**Pattern 1: Sequential Processing**
- Text → Tool → Image → Audio → Response

**Pattern 2: Parallel Processing**
- Text splits to multiple agents simultaneously
- Results combined at the end

**Pattern 3: Conditional Branching**
- If user asks about prices → use price tool → generate city image
- If user asks general question → just respond with text

#### **7. Base64 Encoding**
**Simple Explanation:** A way to represent images as text so they can be transmitted over text-based systems.

**Why It's Used:**
- APIs typically work with text
- Images need to be converted to text format for transmission
- Base64 is a standard encoding method
- Must be decoded back to display as actual image

#### **8. Audio Processing Libraries**
**Simple Explanation:** Specialized code libraries for handling audio:

**PyDub**: A Python library for audio manipulation
- Converting between formats
- Playing audio
- Editing audio files

**AudioSegment**: A component of PyDub for working with audio data
- Creating audio from bytes
- Playing audio in applications

#### **9. Multimodal Agent Workflow Example**
**Simple Explanation:** Step-by-step process in the airline assistant:

1. **User Input**: "Show me ticket prices to London"
2. **Text Processing**: AI understands the request
3. **Tool Execution**: Price lookup tool is called
4. **Image Generation**: DALL-E creates London vacation image
5. **Response Generation**: AI creates text response with price
6. **Audio Generation**: TTS converts response to speech
7. **UI Update**: Interface shows text, image, and plays audio

---

## Advanced Concepts

### **1. Error Handling in Multimodal Systems**
**Simple Explanation:** What to do when different parts of the system fail:

- **Image Generation Fails**: Show text response only
- **Audio Generation Fails**: Display text without speech
- **Tool Call Fails**: Graceful fallback responses
- **Network Issues**: Local caching and retry logic

### **2. Cost Management**
**Simple Explanation:** Multimodal AI is more expensive than text-only:

- **Text Generation**: Fractions of a cent
- **Image Generation**: $0.04 per image
- **Audio Generation**: Moderate cost
- **Strategy**: Use multimodal features strategically, not for every interaction

### **3. User Experience Design**
**Simple Explanation:** Making multimodal AI feel natural:

- **Progressive Enhancement**: Start with text, add multimedia
- **Loading Indicators**: Show users when images/audio are generating
- **Fallback Options**: Always provide text alternatives
- **Performance**: Balance features with speed

### **4. Gradio Advanced UI Components**
**Simple Explanation:** Building more sophisticated interfaces:

**Basic Gradio**: Simple input/output interfaces
**Advanced Gradio**: Custom layouts with multiple components

```python
# Advanced interface with custom layout
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    image_output = gr.Image()
    audio_output = gr.Audio()
    
    # Custom interaction logic
    msg.submit(chat_function, inputs=[msg, chatbot], 
               outputs=[chatbot, image_output, audio_output])
```

---

## Key Technical Skills Developed

### **1. API Management**
- Working with multiple AI service providers
- Handling different authentication methods
- Managing API costs and rate limits
- Error handling and fallback strategies

### **2. User Interface Development**
- Rapid prototyping with Gradio
- Creating professional-looking interfaces
- Implementing streaming responses in UIs
- Building chat interfaces
- Advanced component layouts

### **3. Conversation Management**
- Maintaining context across interactions
- Implementing system prompts effectively
- Using multi-shot prompting techniques
- Dynamic context enhancement

### **4. Tool Integration**
- Defining and implementing custom functions
- Managing tool workflows
- Error handling in tool calls
- Building specialized agents

### **5. Multimodal Development**
- Image generation and processing
- Audio generation and playback
- Coordinating multiple media types
- Base64 encoding/decoding
- File format conversions

### **6. Agent Architecture**
- Designing multi-agent systems
- Coordinating between specialized agents
- Managing agent communication
- Building autonomous workflows

---

## Practical Applications Built

### **1. Company Brochure Generator**
- Web scraping for content
- AI-powered content creation
- Multiple model comparison
- Professional output formatting

### **2. Airline Customer Support**
- Contextual conversations
- Real-time price lookups
- Tool-based functionality
- Professional customer service tone

### **3. Multi-Agent Airline Assistant**
- **Core Functionality**: Flight price lookups
- **Image Generation**: City vacation images using DALL-E 3
- **Audio Generation**: Text-to-speech responses
- **Tool Integration**: Price checking capabilities
- **UI Components**: Chat interface, image display, audio playback
- **Agent Coordination**: Multiple specialized AI components working together

### **4. Multimodal User Experience**
- **Input**: Text-based user queries
- **Processing**: Multi-step agent workflow
- **Output**: Text responses, generated images, spoken audio
- **Interface**: Professional web-based chat application

---

## Learning Progression

**By Day 1:** Multi-API proficiency
**By Day 2:** UI development skills
**By Day 3:** Chatbot creation mastery
**By Day 4:** Tool integration expertise
**By Day 5:** Advanced multimodal agent systems

### **Challenges and Extensions**

#### **Week 2 Challenge Projects:**

1. **Add Booking Tool**
   - Create a "make booking" function
   - Handle booking confirmations
   - File output for booking records

2. **Translation Agent**
   - Use Claude for language translation
   - Display translations alongside original responses
   - Custom Gradio panels for bilingual interface

3. **Audio-to-Text Integration**
   - Speech recognition input
   - Complete audio loop (speech in, speech out)
   - Voice-activated AI assistant

#### **Skills After Completion:**
By the end of Week 2, students have mastered:
- Multi-provider API integration
- Professional UI development
- Advanced conversational AI
- Tool-enhanced AI systems  
- Multimodal AI applications
- Agent-based architectures

**Progress Milestone:** 25% complete toward LLM Engineering mastery

---

## Next Week Preview: Week 3
**Focus**: Open Source AI with Hugging Face
- Hugging Face ecosystem mastery
- Transformers library deep dive
- Tokenizers and model internals
- Running inference on open source models
- Google Colab with GPU acceleration

This progression builds systematically from basic API usage to sophisticated AI systems capable of real business applications, preparing students for the advanced open source development coming in Week 3.
