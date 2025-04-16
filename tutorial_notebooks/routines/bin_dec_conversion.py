import numpy as np

def dec2bin(num, include_fraction=True, fraction_limit=10, bits=None):
    if num == 0:
        return "0".ljust(bits, '0') if bits is not None else "0"

    # Handle negative numbers
    sign = "-" if num < 0 else ""
    num = abs(num)
    
    # Separate the integer and fractional parts of the number
    integer_part = int(num)
    fractional_part = num - integer_part

    # Convert integer part to binary
    binary_integer = bin(integer_part)[2:]  # Use bin() and strip the '0b' prefix

    # Convert fractional part to binary
    binary_fraction = ""
    if include_fraction and fractional_part > 0:
        binary_fraction += "."
        while fractional_part > 0 and len(binary_fraction) <= fraction_limit:
            fractional_part *= 2
            fractional_bit = int(fractional_part)
            binary_fraction += str(fractional_bit)
            fractional_part -= fractional_bit

    # Combine integer and fractional parts
    binary_result = binary_integer + binary_fraction

    # Adjust the result to match the specified number of bits if given
    if bits is not None:
        if include_fraction and '.' in binary_result:
            # Split into integer and fraction parts
            integer_part, fractional_part = binary_result.split('.')
            max_integer_bits = bits - len(fractional_part) - 1  # Account for the decimal point
            integer_part = integer_part.rjust(max_integer_bits, '0')[:max_integer_bits]
            fractional_part = fractional_part.ljust(bits - len(integer_part) - 1, '0')[:bits - len(integer_part) - 1]
            binary_result = integer_part + '.' + fractional_part
        else:
            binary_result = binary_result.rjust(bits, '0')[:bits]

    return sign + binary_result

def bin2dec(binary_str, twos_complement=False):
    # Split the binary string into its integer and fractional parts
    integer_part, _, fraction_part = binary_str.partition('.')
    
    # Determine if the number is negative under two's complement rules
    is_negative = False
    if twos_complement:
        if integer_part:
            # Check if the integer part starts with a 1
            if integer_part[0] == '1':
                is_negative = True
        elif fraction_part and fraction_part[0] == '1':
            # No integer part, but fractional part starts with a 1
            is_negative = True
    
    # Handle two's complement for the integer part if needed
    if is_negative:
        # For integer part
        if integer_part:
            integer_value = int(integer_part, 2)
            integer_decimal = integer_value - (1 << len(integer_part))
        else:
            integer_decimal = 0
        
        # For fractional part, adjust to account for the two's complement
        fraction_value = sum(int(digit) * (2 ** -i) for i, digit in enumerate(fraction_part, start=1))
        fraction_decimal = fraction_value - (2 ** -len(fraction_part))
    else:
        # Convert integer part to decimal
        integer_decimal = int(integer_part, 2) if integer_part else 0

        # Convert fraction part to decimal
        fraction_decimal = sum(int(digit) * (2 ** -i) for i, digit in enumerate(fraction_part, start=1))
    
    # Combine both decimal values
    decimal_number = integer_decimal + fraction_decimal
    
    return decimal_number


def psi2int(psi):
    binn = ""
    for i in psi:
        binn = f"{binn}{i}"
    
    return int(bin2dec(binn))