def get_parking_recommendation(occupancy_percent):
    if occupancy_percent <= 30:
        return "Lot is empty — Go now!"
    elif 30 < occupancy_percent <= 70:
        return "Moderate availability — Proceed with caution"
    elif 70 < occupancy_percent <= 90:
        return "Lot is nearly full — Consider alternatives"
    else:
        return "Lot is full — Avoid!"
