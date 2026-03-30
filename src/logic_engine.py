def get_parking_recommendation(occupancy_percent):
    """
    Logic Architect: Converts percentage to user messages.
    """
    if occupancy_percent <= 30:
        return "Lot is empty — Go now!"
    elif 30 < occupancy_percent <= 70:
        return "Moderate availability — Proceed with caution"
    elif 70 < occupancy_percent <= 90:
        return "Lot is nearly full — Consider alternatives"
    else:
        return "Lot is full — Avoid!"

# Test the logic
# print(get_parking_recommendation(15)) # Output: Lot is empty...
# print(get_parking_recommendation(50)) # Output: Moderate availability — Proceed with caution
# print(get_parking_recommendation(80)) # Output: Lot is nearly full — Consider alternatives
# print(get_parking_recommendation(95)) # Output: Lot is full — Avoid!