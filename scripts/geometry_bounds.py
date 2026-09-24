"""Geometric score certificates verified with exact rational arithmetic.

Public contract: every unknown document x satisfies ||x|| <= R and
intervals[i][0] <= anchors[i] dot x <= intervals[i][1]. Query coordinates,
interval endpoints, R and serialized coefficients denote exact IEEE values.
The caller supplies score/normalization error contracts and state identity.
This module does not authenticate that contract or prove general feasibility.
"""
import hashlib
import json
import math
from fractions import Fraction

SCHEMA = 'geometry-bound-v1'
COEFFICIENT_LIMIT = 1e8


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _fraction(value):
    return Fraction.from_float(float(value))


def _packet(anchors, intervals, query, R, record_id):
    rows = [[float(v) for v in row] for row in anchors]
    limits = [[float(v) for v in pair] for pair in intervals]
    target = [float(v) for v in query]
    radius = float(R)
    if not target or len(target) > 4096 or len(rows) > 512:
        raise ValueError('Unsupported public packet size')
    if len(rows) != len(limits) or any(len(row) != len(target) for row in rows):
        raise ValueError('Query/anchor/interval dimensions differ')
    if any(len(pair) != 2 for pair in limits) or not isinstance(record_id, str):
        raise ValueError('Invalid interval or record identity')
    numbers = target + [radius] + [v for row in rows + limits for v in row]
    if not all(math.isfinite(v) for v in numbers) or radius < 0:
        raise ValueError('Nonfinite input or negative norm bound')
    seen = {}
    for row, (lower, upper) in zip(rows, limits):
        if lower > upper:
            raise ValueError('Inconsistent interval')
        norm_squared = _fraction(radius) ** 2 * sum((_fraction(v) ** 2 for v in row), Fraction())
        if (lower > 0 and _fraction(lower) ** 2 > norm_squared) or (upper < 0 and _fraction(upper) ** 2 > norm_squared):
            raise ValueError('Anchor interval contradicts the norm bound')
        key = tuple(row)
        opposite = tuple(-v for v in row)
        for old, pair in ((seen.get(key), (lower, upper)), (seen.get(opposite), (-upper, -lower))):
            if old is not None and max(old[0], pair[0]) > min(old[1], pair[1]):
                raise ValueError('Duplicate/opposite anchors contradict each other')
        old = seen.get(key, (lower, upper))
        seen[key] = (max(old[0], lower), min(old[1], upper))
    return {'anchors': rows, 'intervals': limits, 'query': target, 'R': radius, 'record_id': record_id}


def _sqrt_upper(value):
    """A rational upward square-root enclosure; no floating-point sqrt trust."""
    if value < 0:
        raise ValueError('Negative exact squared norm')
    if not value:
        return Fraction()
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    precision = max(0, 80 - exponent // 2)
    numerator = value.numerator << (2 * precision)
    root = math.isqrt(numerator // value.denominator)
    if root * root * value.denominator < numerator:
        root += 1
    return Fraction(root, 1 << precision)


def _outward(value, upper):
    try:
        rounded = float(value)
    except OverflowError as exc:
        raise ValueError('Bound cannot be represented by finite binary64') from exc
    if not math.isfinite(rounded):
        raise ValueError('Nonfinite bound')
    represented = _fraction(rounded)
    if (upper and represented < value) or (not upper and represented > value):
        rounded = math.nextafter(rounded, math.inf if upper else -math.inf)
    if not math.isfinite(rounded):
        raise ValueError('Outward bound overflows binary64')
    return rounded


def _certificate(packet, coefficients, packet_digest=None):
    values = [float(v) for v in coefficients]
    if len(values) != len(packet['anchors']) or not all(math.isfinite(v) and abs(v) <= COEFFICIENT_LIMIT for v in values):
        raise ValueError('Invalid coefficient packet')
    # Only the nonzero coefficients need rational vector multiplication.
    residual = [_fraction(v) for v in packet['query']]
    support_lower = support_upper = Fraction()
    for coefficient, row, (lower, upper) in zip(values, packet['anchors'], packet['intervals']):
        if coefficient == 0:
            continue
        weight = _fraction(coefficient)
        support_lower += weight * _fraction(lower if coefficient > 0 else upper)
        support_upper += weight * _fraction(upper if coefficient > 0 else lower)
        residual = [v - weight * _fraction(a) for v, a in zip(residual, row)]
    squared = sum((v * v for v in residual), Fraction())
    slack = _fraction(packet['R']) * _sqrt_upper(squared)
    result = {'schema': SCHEMA, 'packet_sha256': packet_digest or _hash(packet), 'record_id': packet['record_id'],
              'lambda': values, 'lower': _outward(support_lower - slack, False),
              'upper': _outward(support_upper + slack, True)}
    result['certificate_sha256'] = _hash(result)
    return result


def certificate(anchors, intervals, query, R, coefficients, record_id=''):
    """Verify any finite coefficient vector and emit its exact-arithmetic bound."""
    return _certificate(_packet(anchors, intervals, query, R, record_id), coefficients)


def verify_certificate(record, anchors, intervals, query, R, record_id=''):
    """Recompute using only stdlib; this checks binding/integrity, not signatures."""
    try:
        expected = certificate(anchors, intervals, query, R, record['lambda'], record_id)
        return expected == record
    except (ValueError, TypeError, KeyError, OverflowError):
        return False

