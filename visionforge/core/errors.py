"""错误码体系 E100~E500。"""


class VisionForgeError(Exception):
    code = "E000"

    def __init__(self, msg: str = "", code: str | None = None) -> None:
        super().__init__(msg)
        if code:
            self.code = code


class ConfigError(VisionForgeError):
    code = "E100"


class DataError(VisionForgeError):
    code = "E200"


class DescriptorError(VisionForgeError):
    code = "E300"


class ForgeIndexError(VisionForgeError):
    code = "E400"


class PipelineError(VisionForgeError):
    code = "E500"
