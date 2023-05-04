import base64

import google.cloud.aiplatform as vertex_ai


def predict_image(img, endpoint):
    """
    Sends an image payload to the ViT model deployed as a Vertex AI endpoint
    """

    image_b64 = base64.b16encode(img).decode('utf-8')

    return 0
