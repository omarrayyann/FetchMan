from dataclasses import dataclass


@dataclass
class Object:
    body_id: int
    name: str
    category: str
    asset_id: str
    is_static: bool
    has_freejoint: bool

    def position(self, data):
        """World position of the object's body."""
        return data.xpos[self.body_id].copy()

    def quat(self, data):
        """World orientation (wxyz) of the object's body."""
        return data.xquat[self.body_id].copy()
