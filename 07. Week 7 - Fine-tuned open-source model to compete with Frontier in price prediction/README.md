# Complete Guide to LLM Fine-Tuning: From Training to Results

## 1. Training Launch and Monitoring

### Starting the Training Process
When you launch QLoRA fine-tuning, several key things happen:

**Initial Setup:**
- GPU memory usage jumps from ~6GB to ~38GB (nearly maxing out a 40GB A100)
- Training begins with very high loss values that gradually decrease
- The system outputs progress every 50 steps as configured

**Training Timeline:**
- **Example:** 400,000 training samples with batch size 16 across 3-4 epochs
- **Calculation:** (400,000 ÷ 16) × 4 epochs = 100,000 total batch steps
- **Duration:** ~8 hours per epoch on high-end hardware (A100)

### Cost Management Strategies

**Budget-Friendly Training:**
- Use T4 GPU (~$0.10/hour) instead of A100 (~$1.00/hour)
- Reduce dataset size to 20,000-25,000 samples
- Focus on single product category (e.g., appliances only)
- Use batch size of 1-2 on smaller GPUs

**Training Cost Example:**
- Small dataset (25,000 items) on T4 GPU = cents to few dollars
- Full dataset (400,000 items) on A100 GPU = $10-20

## 2. Understanding the Training Process

### The Four Steps of Training

Training happens in cycles, with each cycle containing four crucial steps:

#### Step 1: Forward Pass
**What it is:** Running inference on the model
- Take a training prompt (input text)
- Pass it through the neural network
- Get a prediction for the next token
- **Example:** Input "Price is dollars" → Model predicts "99"

#### Step 2: Loss Calculation
**What it is:** Measuring how wrong the prediction was
- Compare the model's prediction with the actual correct answer
- **Example:** Model predicted "99" but correct answer was "89"
- Calculate a loss score using cross-entropy loss function

#### Step 3: Backward Pass (Backpropagation)
**What it is:** Finding out how to improve
- Look back through the neural network
- Calculate how sensitive the loss is to each parameter
- Determine which weights need adjustment and by how much
- These sensitivities are called "gradients"

#### Step 4: Optimization
**What it is:** Actually updating the model weights
- Use the gradients to adjust the LoRA adapter weights
- Take small steps in the direction that reduces loss
- Step size controlled by learning rate
- Only LoRA adapters are updated, not the base model

### Key Training Concepts Explained

#### Cross-Entropy Loss
**Simple explanation:** The model outputs probabilities for all possible next tokens, not just one prediction.

**How it works:**
1. Model gives probability to each possible token (e.g., "99" = 60%, "89" = 30%, others = 10%)
2. We look at what probability it gave to the correct answer
3. Loss = -log(probability of correct answer)
4. Perfect prediction (100% probability) = 0 loss
5. Worse predictions = higher loss

#### Softmax Function
- Converts raw model outputs (logits) into probabilities
- Ensures all probabilities add up to 1
- Allows us to interpret model confidence

## 3. Monitoring Training with Weights & Biases

### Key Metrics to Watch

#### Training Loss Chart
**What to look for:**
- **Downward trend:** Good! Model is learning
- **Bumpy line:** Normal due to random batches
- **Sudden drops:** Often marks end of epochs (slight overfitting)
- **Plateau below 1.5:** Warning sign of potential overfitting

**Example Pattern:**
```
Epoch 1: Loss drops from ~3.0 to ~2.0
Epoch 2: Small jump up, then drops to ~1.8  
Epoch 3: Small jump up, then drops to ~1.6
Epoch 4+: Performance may degrade (overfitting)
```

#### Learning Rate Schedule
**Cosine Learning Rate:**
- Starts at 0 (warm-up phase)
- Quickly rises to maximum (e.g., 0.0001)
- Slowly decreases in smooth curve
- Ends at 0 when training completes

**Why this works:**
- Warm-up prevents wild initial updates
- High initial rate for major improvements
- Gradual decrease for fine-tuning

#### GPU Utilization
**What to monitor:**
- GPU memory usage (should be high but stable)
- GPU utilization percentage (want close to 100%)
- Power consumption (indicates GPU is working hard)

### Understanding Overfitting

**What happens:**
- Model sees same data multiple times across epochs
- Starts memorizing specific examples rather than learning patterns
- Training loss keeps dropping, but real performance degrades

**Signs of overfitting:**
- Loss drops below ~1.5
- Sudden improvements at epoch boundaries
- Model performs worse on new data despite lower training loss

**Solution:**
- Stop training after 2-3 epochs
- Use validation dataset to monitor real performance
- Save model checkpoints to pick best performing version

## 4. Model Saving and Checkpointing

### Hugging Face Hub Integration

**Automatic Saving:**
- Model saves to Hugging Face Hub every 5,000 steps (configurable)
- Each save creates a new commit/revision
- Can load any checkpoint for testing

**File Structure:**
- `adapter_model.safetensors`: Main LoRA weights (~109MB for R=32)
- `adapter_config.json`: Configuration (R value, target modules, etc.)
- Training metadata and logs

**Benefits:**
- Test models before training completes
- Compare different training stages
- Resume training if interrupted
- Select best-performing checkpoint

## 5. Advanced Training Concepts

### Batch Processing
**How it works:**
- Process multiple examples simultaneously (e.g., 16 at once)
- More efficient GPU utilization
- Provides more stable gradient estimates

**Memory considerations:**
- Higher batch size = more GPU memory needed
- Gradient accumulation allows effective larger batches with less memory

### Learning Rate Scheduling
**Why it matters:**
- Too high: Model bounces around, unstable learning
- Too low: Very slow learning, may get stuck
- Changing rate during training optimizes both speed and stability

### Dropout Regularization
**What it does:**
- Randomly sets 10% of neurons to zero during training
- Prevents overfitting by forcing model to use different pathways
- Model becomes more robust and generalizable

## 6. Model Inference and Evaluation

### Loading Fine-tuned Models

**Process:**
1. Load base model (LLaMA 3.1) with quantization
2. Load LoRA adapters as PEFT model
3. Combine base model with adapters

**Memory usage:**
- Base model: ~5.6GB (4-bit quantized)
- With LoRA adapters: ~5.7GB (additional 109MB)

### Inference Strategies

#### Standard Approach
- Take highest probability token as prediction
- Simple but may miss nuanced predictions

#### Improved Approach
- Consider top 3 most likely tokens
- Take weighted average of their values
- Better for numerical predictions like prices
- Treats regression-like problems more appropriately

### Evaluation Process

**Test Setup:**
- Use held-out test dataset (never seen during training)
- Only provide text description, not the price
- Compare predictions to actual prices
- Calculate mean absolute error

**Key Considerations:**
- Product prices have natural volatility
- Model doesn't know about sales, discounts, or market changes
- Some variation is unavoidable and normal

## 7. Results and Performance

### Performance Comparison
Based on the actual training results:

| Model Type | Error (USD) | Notes |
|------------|-------------|-------|
| Constant (average) | 146 | Baseline benchmark |
| Traditional ML | 139 | Basic features |
| Random Forest | 97 | Advanced ML |
| Human Performance | 127 | Manual prediction |
| GPT-4 | 76 | Frontier model |
| **Fine-tuned LLaMA 3.1** | **47** | **Best performance** |

**Key Insights:**
- Fine-tuned open source model beats GPT-4 (trillion+ parameters)
- Specialized training outperforms general-purpose models
- 8 billion parameter model with task-specific training > massive general model

### Why Fine-tuning Works So Well

**Specialization Advantage:**
- Model learns specific patterns in product pricing
- Understands relationships between product features and prices
- Optimized for single task vs. general capabilities

**Data Efficiency:**
- Focused training data (all price predictions)
- No distraction from unrelated tasks
- Every training example reinforces the same skill

## 8. Hyperparameter Optimization Opportunities

### Areas for Improvement

**Training Parameters:**
- Learning rate (try 0.00001 to 0.001)
- Batch size (experiment with hardware limits)
- Number of epochs (2-4 typically optimal)
- Dropout rate (0.05 to 0.2)

**LoRA Configuration:**
- R value (8, 16, 32, 64)
- Alpha scaling (try Alpha = R or Alpha = 2×R)
- Target modules (experiment with different layer combinations)

**Data Optimization:**
- Prompt engineering (different ways to present product info)
- Data curation (filtering, cleaning, augmentation)
- Feature engineering (how to describe products)

**Model Selection:**
- Try different base models (Qwen, Phi-3, different LLaMA sizes)
- Experiment with 8-bit vs 4-bit quantization
- Test larger models (14B parameters)

### Challenge Goals
- **Target:** Get below $40 error
- **Methods:** Systematic hyperparameter tuning
- **Tools:** Weights & Biases for experiment tracking
- **Approach:** Scientific methodology with controlled variables

## 9. Production Considerations

### Deployment Readiness
After successful fine-tuning, the model is ready for:
- API deployment for production use
- Integration into applications
- Real-time price prediction services
- Scalable inference infrastructure

### Business Impact
- **Cost savings:** No API fees for inference
- **Customization:** Tailored to specific business needs
- **Control:** Full ownership of model and improvements
- **Performance:** Beats expensive frontier models for specific tasks

This comprehensive guide covers the complete journey from launching training to achieving state-of-the-art results with fine-tuned open source models.
