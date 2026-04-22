import asyncio
import aiohttp
import uuid
import json
import sys
import os

request_queue = []


async def fetch_data(session, url, use_json=True):
    print(f"Starting task: {url}")
    async with session.get(url) as response:
        if use_json:
            data = await response.json()
            print(f"Completed task: {url}")
            return data
        data = await response.text()
        print(f"Completed task: {url}")
        return data


def add_request(url, callback, error_callback, use_json=True):
    request_queue.append({
        "url": url,
        "callback": callback,
        "error_callback": error_callback,
        "use_json": use_json
    })


async def fetch_data_callback(session, url, callback, error, use_json=True):
    try:
        res = await fetch_data(session, url, use_json=use_json)
        callback(res)
    except Exception as e:
        def err_fn(e): return print(e)
        if (error is not None):
            err_fn = error
        err_fn(e)


async def run_until_complete():
    tasks = []
    async with aiohttp.ClientSession() as session:
        # weird to repeat the loop twice but it works!
        while (len(request_queue) > 0):
            tasks = []
            while (len(request_queue) > 0):
                t = request_queue.pop()
                use_json = True
                if "use_json" in t.keys():
                    use_json = t.get("use_json")
                tasks.append(
                    fetch_data_callback(
                        session, t['url'],
                        t['callback'], t['error_callback'],
                        use_json=use_json
                    )
                )
            await asyncio.gather(*tasks)


query = "license:mit"
query_string = f"q={query}"
url = f"https://api.github.com/search/repositories?{query_string}"

output_path = "scrape_test"


def make_type(path):
    return {
        "path": path,
        "names": set()
    }


download_types = {
    "py": make_type(""),
    "js": make_type(""),
    "ts": make_type(""),
    "cs": make_type(""),
    "lua": make_type(""),
    "rs": make_type(""),
    "md": make_type(""),
    "go": make_type(""),
    # TODO other types of code !
}


def make_download_handler(name):
    file_save_path = os.path.join(output_path, str(uuid.uuid4()) + "-" + name)

    def handle_download(result):
        with open(file_save_path, "w+") as f:
            f.write(result)
    return handle_download


def parse_contents_res(result):
    for path in result:
        if "download_url" in path and path["download_url"]:
            file_type = path["name"].split(".")[-1]
            print(file_type)
            if file_type not in download_types.keys():
                print(f"File type {file_type}; not in list, skipping!!!!")
                continue
            # download the file
            print("DOWNLOAD!!")
            add_request(
                path["download_url"],
                make_download_handler(path["name"]),
                None,
                use_json=False
            )

        else:
            add_request(
                path["url"],
                parse_contents_res,
                None
            )


def parse_file_res(result):
    print("NEXT FILE REQEST")
    print(json.dumps(result, indent=2))
    for item in result:
        print(item["name"])
        url: str = item["contents_url"]
        print(url)
        add_request(
            url,
            parse_contents_res,
            None
        )


def parse_list_res(result):
    print(json.dumps(result, indent=2))
    for item in list(result["items"]):
        print(item["name"])
        url: str = item["contents_url"]
        url = url.removesuffix("{+path}")
        print(url)
        add_request(
            url,
            parse_contents_res,
            None
        )


add_request(
    url, parse_list_res, None
)
if __name__ == '__main__':
    asyncio.run(run_until_complete())
