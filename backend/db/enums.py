import enum


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"

class SubscriptionPlan(str, enum.Enum):
    free = "free"
    basic = "basic"
    pro = "pro"
    enterprise = "enterprise"

class UserRole(enum.Enum):
    user = "user"
    admin = "admin"

class UserStatus(str, enum.Enum):
    active = "active"
    banned = "banned"

