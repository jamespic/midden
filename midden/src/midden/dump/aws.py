import argparse
import base64
import json
import uuid
from subprocess import run

import boto3

from midden.dump.inject import _TARBALL


def build_presigned_s3_put_url(
    bucket_name, object_name, expiration=3600, region_name="us-east-1"
):
    """Generate a presigned URL to upload an S3 object"""
    # Generate a presigned URL for the S3 object
    s3 = boto3.client("s3", endpoint_url=f"https://s3.{region_name}.amazonaws.com")
    response = s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={
            "Bucket": bucket_name,
            "Key": object_name,
            "ContentType": "application/octet-stream",
        },
        ExpiresIn=expiration,
    )
    return response


def build_injection_script(bucket_name, object_name, region_name="us-east-1"):
    """Build a shell script that dumps the heaps of all Python processes and uploads the dump to S3 using a presigned URL."""
    presigned_url = build_presigned_s3_put_url(
        bucket_name, object_name, region_name=region_name
    )
    return f"""/bin/sh -c "
    mkdir -p /tmp/midden_injection
    cd /tmp/midden_injection
    echo '{base64.encodebytes(_TARBALL).decode()}' | base64 -d | tar -xz
    python3 inject.py --upload-url '{presigned_url}' all
    "
    """


def inject_into_ecs_container(
    cluster: str,
    container: str,
    task: str,
    bucket_name: str,
    object_name: str,
    region_name="us-east-1",
):
    """Inject the midden code into a running ECS container and dump the heaps of all Python processes."""
    try:
        # Check session-manager-plugin installed
        run(["session-manager-plugin", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "session-manager-plugin is not installed or not found in PATH. Please install it to use this feature."
        )
    injection_script = build_injection_script(bucket_name, object_name, region_name)
    ecs_client = boto3.client("ecs", region_name=region_name)
    result = ecs_client.execute_command(
        cluster=cluster,
        container=container,
        command=injection_script,
        interactive=True,
        task=task,
    )
    session = result["session"]
    run(
        ["session-manager-plugin", json.dumps(session), region_name, "StartSession"],
        check=True,
    )


def main():
    arg_parser = argparse.ArgumentParser(
        description="Inject midden into a running ECS container and dump the heaps of all Python processes."
    )
    arg_parser.add_argument("--cluster", required=True, help="ECS cluster name")
    arg_parser.add_argument("--container", required=True, help="ECS container name")
    arg_parser.add_argument("--task", required=True, help="ECS task ID")
    arg_parser.add_argument(
        "--bucket", required=True, help="S3 bucket name to upload the dump"
    )
    arg_parser.add_argument(
        "--object",
        required=False,
        help="S3 object name for the dump file. If no object name is provided, a random name will be used",
        default=None,
    )
    arg_parser.add_argument(
        "--region",
        required=False,
        help="AWS region for the S3 bucket. Default is us-east-1",
        default="us-east-1",
    )
    arg_parser.add_argument(
        "--output-file",
        "-o",
        required=False,
        help="Local file path to save the dump file. If not provided, the dump will be uploaded to S3 only.",
        default=None,
    )
    args = arg_parser.parse_args()

    if args.object is None:
        args.object = f"midden_dump_{uuid.uuid4()}.jsonl"

    inj_result = inject_into_ecs_container(
        cluster=args.cluster,
        container=args.container,
        task=args.task,
        bucket_name=args.bucket,
        object_name=args.object,
        region_name=args.region,
    )
    print(f"Injection result: {inj_result}")

    if args.output_file is not None:
        # Download the dump file from S3 to the local file system
        s3_client = boto3.client("s3", region_name=args.region)
        s3_client.download_file(args.bucket, args.object, args.output_file)
        print(f"Dump file downloaded to {args.output_file}")


if __name__ == "__main__":
    main()
