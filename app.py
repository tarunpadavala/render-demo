import os
import numpy as np
import torch
import xgboost as xgb
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.wsgi import get_wsgi_application

# ✅ Ensure Django is properly set up
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fakeprofile.settings")
application = get_wsgi_application()  # ✅ Gunicorn looks for `application`

# ✅ Load PyTorch Model Once (Globally)
MODEL_PATH = os.path.join(os.getcwd(), "xg_model.ckpt")

try:
    model = torch.load(MODEL_PATH, weights_only=False)
    model.eval()
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@csrf_exempt
def predict_twitter(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            input_features = np.array([[
                int(data['sex_code']), int(data['statuses_count']), int(data['followers_count']),
                int(data['friends_count']), int(data['favourites_count']), int(data['listed_count']),
                int(data['lang_code'])
            ]])
            input_tensor = torch.tensor(input_features, dtype=torch.float32)

            if model is None:
                return JsonResponse({"error": "Model failed to load"}, status=500)

            with torch.no_grad():
                prediction = model(input_tensor)

            result = int(torch.round(prediction).item())

            return JsonResponse({'prediction': "Fake" if result == 1 else "Not Fake"})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)
