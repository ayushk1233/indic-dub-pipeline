from pydantic import BaseModel


class BundleMetadata(BaseModel):
    bundle_version: str = "1.0"
    job_id: str
    created_at: str


class BundlePaths(BaseModel):
    request_json: str
    reference_audio: str


class BundleManifest(BaseModel):
    metadata: BundleMetadata
    paths: BundlePaths
