import re


def mask_phone(phone: str) -> str:
    """
    手机号脱敏
    
    保留前3位和后4位，中间用****替换
    
    Args:
        phone: 手机号字符串
    
    Returns:
        str: 脱敏后的手机号
    """
    if not phone:
        return phone
    phone = re.sub(r"[^0-9]", "", phone)
    if len(phone) >= 11:
        return phone[:3] + "****" + phone[-4:]
    elif len(phone) >= 7:
        return phone[:3] + "****" + phone[-2:]
    return phone


def mask_id_card(id_card: str) -> str:
    """
    身份证号脱敏
    
    保留前6位（地址码）和后4位（顺序码+校验码），中间用*替换
    支持15位和18位身份证号，保留末尾的X/x字符
    
    Args:
        id_card: 身份证号字符串
    
    Returns:
        str: 脱敏后的身份证号
    """
    if not id_card:
        return id_card
    id_card = id_card.upper()
    cleaned = re.sub(r"[^0-9X]", "", id_card)
    if len(cleaned) >= 18:
        return cleaned[:6] + "**********" + cleaned[-4:]
    elif len(cleaned) >= 15:
        return cleaned[:6] + "*******" + cleaned[-3:]
    return id_card


def mask_name(name: str) -> str:
    """
    姓名脱敏
    
    单字姓名保持不变，双字姓名保留首字，多字姓名保留首尾字
    
    Args:
        name: 姓名字符串
    
    Returns:
        str: 脱敏后的姓名
    """
    if not name:
        return name
    if len(name) == 1:
        return name
    elif len(name) == 2:
        return name[0] + "*"
    else:
        return name[0] + "*" * (len(name) - 2) + name[-1]


def mask_email(email: str) -> str:
    """
    邮箱脱敏
    
    保留@前部分的前2位和后1位，中间用***替换
    
    Args:
        email: 邮箱字符串
    
    Returns:
        str: 脱敏后的邮箱
    """
    if not email:
        return email
    match = re.match(r"^(.+?)@(.+)$", email)
    if not match:
        return email
    local = match.group(1)
    domain = match.group(2)
    if len(local) <= 2:
        return local + "***@" + domain
    return local[:2] + "***" + local[-1] + "@" + domain if len(local) > 3 else local[:2] + "***@" + domain


def mask_bank_card(bank_card: str) -> str:
    """
    银行卡号脱敏
    
    保留前4位和后4位，中间用****替换
    
    Args:
        bank_card: 银行卡号字符串
    
    Returns:
        str: 脱敏后的银行卡号
    """
    if not bank_card:
        return bank_card
    bank_card = re.sub(r"[^0-9]", "", bank_card)
    if len(bank_card) >= 16:
        return bank_card[:4] + "**** **** ****" + bank_card[-4:]
    elif len(bank_card) >= 10:
        return bank_card[:4] + "****" + bank_card[-4:]
    return bank_card


def mask_address(address: str) -> str:
    """
    地址脱敏
    
    保留前4位和后2位，中间用**替换
    
    Args:
        address: 地址字符串
    
    Returns:
        str: 脱敏后的地址
    """
    if not address:
        return address
    if len(address) <= 4:
        return address[:2] + "**"
    return address[:4] + "**" + address[-2:] if len(address) > 6 else address[:4] + "**"


def auto_mask(text: str) -> str:
    """
    自动脱敏文本中的敏感信息
    
    自动识别并脱敏文本中的手机号和身份证号
    
    Args:
        text: 原始文本
    
    Returns:
        str: 脱敏后的文本
    """
    if not text:
        return text
    text = re.sub(r"(\d{3})\d{4}(\d{4})", r"\1****\2", text)
    text = re.sub(r"(\d{6})\d{8,10}(\d{4})", r"\1**********\2", text)
    text = re.sub(r"(\d{6})\d{7}(\d{3})", r"\1*******\2", text)
    return text
