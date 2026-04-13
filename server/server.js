require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { HfInference } = require('@huggingface/inference');

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: 'uploads/' });

// Initialize Hugging Face Inference
// Make sure you have HF_TOKEN set in your environment variables
const hf = new HfInference(process.env.HF_TOKEN);

app.post('/api/upload', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  // In a real implementation we would parse the PDF/TXT, generate embeddings,
  // and store them in a vector database here. For now, since it's a demo, we
  // just acknowledge the upload.
  res.json({ message: 'File uploaded successfully', filename: req.file.originalname });
});

app.post('/api/chat', async (req, res) => {
  const { message } = req.body;
  if (!message) {
    return res.status(400).json({ error: 'Message is required' });
  }

  try {
    // Basic text generation using Hugging Face Inference
    // Using Qwen Finance 7b as requested: "Himanshu2124/qwen-finance-7b"
    // Since this might not be available as a free inference endpoint, this may throw
    // if the model is too large or requires Pro. We will fall back gracefully or
    // just pass the model string as is.
    
    // As a robust alternative for demo if qwen-finance-7b fails, we could use Mistral
    // but we will try the requested model.
    let responseText = "";
    try {
      const completion = await hf.textGeneration({
        model: 'Himanshu2124/qwen-finance-7b',
        inputs: message,
        parameters: { max_new_tokens: 256, temperature: 0.7 }
      });
      responseText = completion.generated_text;
    } catch (e) {
      console.error("HF Inference Error, returning mock response. Error: ", e.message);
      responseText = "This is a mock response from qwen-finance-7b because the model might not be loaded on the Hugging Face Inference API. Simulated response: You asked about " + message;
    }

    res.json({ response: responseText });
  } catch (error) {
    console.error("Server error:", error);
    res.status(500).json({ error: 'Failed to generate response' });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
