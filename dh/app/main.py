from util import allowed_file, batch_inference, parse_es_results, es_query_str, es_query, load_image
from config import Config as cfg
import requests
from flask import Flask, request, jsonify, render_template
import base64
import imghdr
import time

from config import logger

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = cfg.MAX_UPLOAD_SIZE * 1024
app.config['JSON_AS_ASCII'] = False


@app.route('/', methods=['POST', 'GET'])
def search():
    """
    Perform query and visualize results
    :return:
    """
    images = []
    query_image = ""
    error = ""
    useURL = 0
    querytime = 0
    max_results = cfg.ES_MAX_RESULTS
    perform_search = False

    try:
        # Query image from results
        if request.method == 'GET':
            if "imagePath" in request.values.keys():
                try:
                    image_path = request.args["imagePath"]
                    max_results = request.args["maxResults"]
                except:
                    error = "Error in form data"
                    raise
                try:
                    max_results = int(max_results)
                except:
                    error = "Maximum results is not a number"
                    raise
                try:
                    if cfg.LOCAL_IMAGES:
                        local_file = open("/" + image_path, "rb")
                        img = local_file.read()
                    else:
                        url = image_path
                        r = requests.get(url, stream=True, verify=False)
                        img = r.content
                    perform_search = True
                except:
                    error = "Invalid URL for query image"
                    raise
                try:
                    ext = imghdr.what("", h=img)
                except:
                    error = "Query image is not a valid image file"
                    raise

        if request.method == 'POST':
            perform_search = True
            try:
                useURL = request.values["useURL"]
                url = request.values["url"]
                max_results = request.values["maxResults"]
            except:
                error = "Error in form data"
                raise
            try:
                max_results = int(max_results)
            except:
                error = "Maximum results is not a number"
                raise
            # Query image from URL
            if useURL == "1":
                if (url == ""):
                    error = "URL is empty"
                    raise Exception(error)
                else:
                    try:
                        r = requests.get(url, stream=True)
                        img = r.content
                    except:
                        error = "Invalid URL for query image"
                        raise
            elif "upload" in request.files and request.files["upload"]:
                # Query image uploaded
                file = request.files["upload"]
                if not file.filename:
                    error = "File type is not accepted"
                    raise Exception(error)
                else:
                    try:
                        img = file.read()
                    except:
                        error = "Could not read file"
                        raise
            else:
                # Use previous query image
                try:
                    b64_string = request.form.get("previous_query")
                    img = base64.decodebytes(b64_string)
                except:
                    error = "No file uploaded"
                    raise
            try:
                ext = imghdr.what("", h=img)
            except:
                error = "Not a valid image file"
                raise

        if perform_search:
            start = time.time()
            try:
                query_image = f'data:image/{ext};base64,' + base64.b64encode(img).decode('utf-8')
            except:
                error = "Could not read file"
                raise
            try:
                img = load_image(img)
                files = {"query_image": {"qi.jpg": img}}
                code_dict = batch_inference(files)[0]['codes']
                q = es_query_str(code_dict)

                res = es_query(q, max_results)
            except:
                error = "No result from Elasticsearch"
                raise
            try:
                images = parse_es_results(res)
            except:
                error = "Could not parse Elasticsearch results"
                raise

            querytime = time.time() - start

    except Exception as e:
        logger.error(e, exc_info=True)

    return render_template('index.html', useURL=useURL, images=images, query_image=query_image, error=error,
                           time=querytime, display_welcome=not perform_search, max_results=max_results,
                           thumbs_url=cfg.THUMBS_URL, images_url=cfg.IMAGES_URL)


@app.route('/infer/', methods=['POST', 'GET'])
def infer():
    """
    Receive images, process them and return JSON
    :return:
    """
    if request.method == 'POST':
        files = {}
        response = []
        for key, file in request.files.items():
            if file and allowed_file(file.filename):
                buf = file.read()
                img = load_image(buf)
                files[key] = {file.filename: img}
            else:
                response += [{"status_code": "ERROR", "status_msg": "wrong file type", 'filename': file.filename,
                              'fieldname': key}]
        response += [batch_inference(files)]
        return jsonify({"results": response}), 201
    else:
        s_title = "Binary Hashcode Computation"
        s_version = "0.3"
        return """
    <!doctype html>
    <title>%s</title>
    <h2>%s</h2>
    <I>Version: %s</I>
    <form action="#" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
      <br /><br /><input type=submit value="Compute code">
    </form>
    <hr>
    <p></p>
    """ % (s_title, s_title, s_version)


if __name__ == '__main__':
    app.run(threaded=True)
