__doc__ = "GitHub Actions script to download a GitHub artifact from a repo."

import argparse
import datetime
from functools import partial
import json
import os
import subprocess
import sys
import zipfile

# program name
_PROGNAME = __file__.replace("\\", "/").split("/")[-1]
# argparse argument help strings
_HELP_REPO = "Name of the target repository, in the format owner/repo_name"
_HELP_NAME = """\
Name of the artifact to download. If -i/--id is specified, the value of this
option is ignored (even if there is no artifact that has this name).\
"""
_HELP_ID = """\
ID of the artifact to download. If not specified, then if there are multiple
artifacts with the same name, the latest one will be downloaded. Corresponds to
the \"id\" key in the JSON object describing an artifact. See
https://docs.github.com/en/rest/reference/actions for more details.\
"""
_HELP_DOWNLOAD_DIR = """\
Directory to download the artifact to. Defaults to the current directory.\
"""
_HELP_UNZIP = """\
Specify to also unzip the downloaded artifact files to the directory specified
by -d/--download-dir after the artifact has been downloaded.\
"""
_HELP_TOKEN = """\
GitHub authentication token. This is not required if the GITHUB_TOKEN
environment variable is set.\
"""


def datetime_created_at(artifact):
    """Returns ``created_at`` key of the JSON artifact object as ``datetime``.

    :param artifact: JSON object representing GitHub artifact
    :type artifact: dict
    :rtype: :class:`datetime.datetime`
    """
    return datetime.datetime.strptime(
        artifact["created_at"], "%Y-%m-%dT%H:%M:%SZ"
    )


def main(args = None):
    """Main method called as entry point.

    :param args: List of strings to pass to :class:`argparse.ArgumentParser`.
    :type args: list
    :returns: Exit value
    :rtype: int
    """
    # instantiate ArgumentParse and add arguments. help width is set to 80 cols
    # although we are technically using the private argparse API.
    arp = argparse.ArgumentParser(
        prog = _PROGNAME,
        formatter_class = partial(
            argparse.RawDescriptionHelpFormatter, width = 80
        )
    )
    arp.add_argument("repo", help = _HELP_REPO)
    arp.add_argument("-n", "--name", help = _HELP_NAME)
    arp.add_argument("-i", "--id", type = int, help = _HELP_ID)
    arp.add_argument(
        "-d", "--download-dir", default = ".", help = _HELP_DOWNLOAD_DIR
    )
    arp.add_argument(
        "-u", "--unzip", action = "store_true", help = _HELP_UNZIP
    )
    arp.add_argument(
        "-t", "--token", default = (
            None if "GITHUB_TOKEN" not in os.environ
            else os.environ["GITHUB_TOKEN"]
        ), help = _HELP_TOKEN
    )
    # parse arguments
    args = arp.parse_args(args = args)
    # base curl command to use
    curl_cmd = (
        "curl -H \"Accept: application/vnd.github.v3+json\" "
        f"https://api.github.com/repos/{args.repo}/actions/artifacts"
    )
    # other arguments to pass to subprocess.run
    run_args = dict(stderr = subprocess.PIPE, stdout = subprocess.PIPE)
    # if id is not None, then look for the artifact with the specific ID
    if args.id is not None:
        curl_res = subprocess.run(
            (curl_cmd + f"/{args.id}").split(), **run_args
        )
    # else return all the artifacts
    else:
        curl_res = subprocess.run(curl_cmd.split(), **run_args)
    # load result from curl_res.stdout
    artifacts = json.loads(curl_res.stdout)
    # download URL for the artifact
    download_url = None
    # if artifacts has "id" key, then this is an individual result
    if "id" in artifacts:
        # get download URL and overwrite artifact name
        download_url = artifacts["archive_download_url"]
        args.name = artifacts["name"]
    # else if artifacts has the "artifacts" key, we got a bunch of results
    elif "artifacts" in artifacts:
        # get just the artifacts
        artifacts = artifacts["artifacts"]
        # sort by "created_at" key using datetime_created_at, latest first
        artifacts.sort(key = datetime_created_at, reverse = True)
        # get latest version of artifact with name artifact that's not expired
        for artifact in artifacts:
            if artifact["name"] == args.name and not artifact["expired"]:
                download_url = artifact["archive_download_url"]
                break
    # if download_url is None, error
    if download_url is None:
        print(
            f"{_PROGNAME}: error: couldn't find artifact {args.name} in "
            f"repo {args.repo} (id={args.id})",
            file = sys.stderr
        )
        return 1
    # set args.id to the ID in the download URL
    args.id = download_url.split("/")[-2]
    # wget command to download artifact
    wget_cmd = (
        f"wget --header \"Authorization: token {args.token}\" "
        f"--output-document {args.download_dir}/{args.name}.zip {download_url}"
    )
    # download artifact
    wget_res = subprocess.run(wget_cmd.split(), **run_args)
    # if error code != 0, there was an error
    if (wget_res.returncode != 0): 
        print(
            f"{_PROGNAME}: error: wget couldn't download artifact {args.name} "
            f"in repo {args.repo} (id={args.id}) to target "
            f"{args.download_dir}/{args.name}.zip"
        )
        return wget_res.returncode
    # if downloaded and -u/--unzip is passed, unzip the file
    with zipfile.ZipFile(f"{args.download_dir}/{args.name}.zip") as zf:
        zf.extractall(path = args.download_dir)


if __name__ == "__main__":
    sys.exit(main())