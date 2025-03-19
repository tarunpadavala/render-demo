import os
import numpy as np
import torch
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Load PyTorch Model Once (Globally to Avoid Reloading Each Request)
MODEL_PATH = os.path.join(os.getcwd(), "xg_model.ckpt")

try:
    model = torch.load(MODEL_PATH, weights_only=False)  # ✅ Fix: Ensure full model loads
    model.eval()  # Set model to evaluation mode
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@csrf_exempt
def predict_twitter(request):
    if request.method == 'POST':
        try:
            # ✅ Fix: Use `json.loads(request.body)` to parse JSON requests
            data = json.loads(request.body)

            # Extract input features
            input_features = np.array([[
                int(data['sex_code']), int(data['statuses_count']), int(data['followers_count']),
                int(data['friends_count']), int(data['favourites_count']), int(data['listed_count']),
                int(data['lang_code'])
            ]])

            # ✅ Fix: Convert NumPy array to PyTorch tensor
            input_tensor = torch.tensor(input_features, dtype=torch.float32)

            # ✅ Fix: Ensure model is loaded before inference
            if model is None:
                return JsonResponse({"error": "Model failed to load"}, status=500)

            # Make Prediction
            with torch.no_grad():  # Disable gradient calculation for inference
                prediction = model(input_tensor)

            # Convert Prediction to Binary (0 or 1)
            result = int(torch.round(prediction).item())

            # Return JSON response
            return JsonResponse({'prediction': "Fake" if result == 1 else "Not Fake"})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)
