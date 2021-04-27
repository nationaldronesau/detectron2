import hashlib
import hmac
from requests import Request
from requests.auth import AuthBase

from config.constants import SMARTDATA_API_KEY


def compute_signature(endpoint: str, method: str, body: str) -> hmac.HMAC:
    """
    Computes the signature for a signed request to smartdata, uses the DETECTION_SERVICE_API_KEY environment variable
    :param endpoint:
    :param method:
    :param body:
    :return:
    @author: Conor Brosnan <c.brosnan@nationaldrones.com>
    """
    params = [endpoint, method, body, SMARTDATA_API_KEY]
    data = "".join(params)
    computed_sig = hmac.new(SMARTDATA_API_KEY.encode('utf-8'), data.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()
    return computed_sig


class SignatureAuth(AuthBase):
    """
    Adds a signature to a celery request for sending to smartdata
    @author: Conor Brosnan <c.brosnan@nationaldrones.com>
    """

    def __call__(self, request) -> Request:
        if request.body is None:
            body = ""
        else:
            body = request.body

        request.headers = {
            'Accept': 'application/json',
            'Signature': compute_signature(request.path_url, request.method, body),
            'Content-Type': 'application/json',
        }
        return request
