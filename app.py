import os
import numpy as np
import pickle  # Using pickle instead of joblib
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.wsgi import get_wsgi_application

# Set up Django application manually (needed for Render)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fakeprofile.settings")
application = get_wsgi_application()

@csrf_exempt  # Disable CSRF for direct API calls
def predict_twitter(request):
    if request.method == 'POST':
        try:
            # Extract form data from POST request
            sex_code = int(request.POST.get('sex_code'))
            statuses_count = int(request.POST.get('statuses_count'))
            followers_count = int(request.POST.get('followers_count'))
            friends_count = int(request.POST.get('friends_count'))
            favourites_count = int(request.POST.get('favourites_count'))
            listed_count = int(request.POST.get('listed_count'))
            lang_code = int(request.POST.get('lang_code'))  # Ensure this is an integer

            # Convert input data to NumPy array
            input_features = np.array([[
                sex_code, statuses_count, followers_count,
                friends_count, favourites_count, listed_count, lang_code
            ]])

            # Load the ML model (Ensure correct path on Render)
            model_path = os.path.join(os.getcwd(), "model.pkl")

            if not os.path.exists(model_path):
                return JsonResponse({"error": "Model file not found!"}, status=500)

            # Load model using pickle (compatible with old working code)
            with open(model_path, 'rb') as file:
                model = pickle.load(file)

            # Make prediction
            prediction = model.predict(input_features)
            result = int(prediction[0])

            # Send JSON response
            return JsonResponse({'prediction': "Fake" if result == 1 else "Not Fake"})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)
