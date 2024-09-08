import numpy as np
from fastapi import FastAPI, UploadFile, File, Request, Response, HTTPException
import uvicorn
import io
import cv2

app = FastAPI()

@app.post('/translate_photo')
def process_audio(file: UploadFile, lang_from: str = 'en', lang_to: str = 'ru') -> Response:
    # Check if the uploaded file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File is not an image")

    img_bytes = io.BytesIO(file.file.read()).read()
    image = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_COLOR)


    return Response(content=cv2.imencode('.png', image)[1].tobytes(),
                    status_code=200, media_type='image/png')

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8098, workers=1, reload=False)