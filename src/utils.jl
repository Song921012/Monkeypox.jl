function pairrate!(t, p0, pend, r)
    return p0 + (pend - p0) * exp(-r * t)
end