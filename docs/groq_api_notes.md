# Groq API Notes

Source: https://console.groq.com/docs/vision (accessed during implementation)

The official Groq vision documentation currently identifies `qwen/qwen3.6-27b` as a multimodal model supporting text and image inputs, chat completions, JSON mode and tool use. The documented API uses the `chat.completions` endpoint with a user message whose content is an array containing a text item and an `image_url` item. Locally stored images are sent as a `data:image/jpeg;base64,...` URL. The documentation states a maximum image URL request size of 20 MB and up to five images per request for that model.

The implementation must treat Groq as an external multimodal analysis layer, not as ARQTECH, YOLO, ground truth, robot controller, geometry engine or training target. The API key must remain in Streamlit Secrets as `GROQ_API_KEY` or a runtime environment variable and must never be stored in source, logs, screenshots, reports or Experience Memory.
