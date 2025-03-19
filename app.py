import os
import numpy as np
import torch
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Load PyTorch Model Once (Globally to Avoid Reloading Each Request)
MODEL_PATH = os.path.join(os.getcwd(), "xg_model.ckpt")
model = torch.load(MODEL_PATH)
model.eval()  # Set model to evaluation mode

@csrf_exempt
def predict_twitter(request):
    if request.method == 'POST':
        try:
            # Extract form data
            data = request.POST
            input_features = np.array([[
                int(data['sex_code']), int(data['statuses_count']), int(data['followers_count']),
                int(data['friends_count']), int(data['favourites_count']), int(data['listed_count']),
                int(data['lang_code'])
            ]])

            # Convert to PyTorch Tensor
            input_tensor = torch.tensor(input_features, dtype=torch.float32)

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
