import requests
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create ES index with all possible d1 and d2 neighbors for 16 bit codes')

    parser.add_argument(
        '--es_url',
        default="http://elasticsearch:9200",
        type=str,
        help='Elastic Search URL with port (default: http://elasticsearch:9200)'
    )

    args = parser.parse_args()
    es_url = args.es_url

    hdist = ' {    ' \
            '    "script": {     ' \
            '       "lang": "painless", "source": "' \
            '64-Long.bitCount(params.subcode^doc[params.field].value)" }}'
    r = requests.post(es_url + "/_scripts/hd64", hdist, headers={'Content-Type': 'application/json'})
    print(r.text)
