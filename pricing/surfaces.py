class BondSurface:
    def __init__(self, name):
        self.name = name

    def import_data(self, data_frame):
        data_frame = data_frame[["Issued Amount", "Outstanding", "Coupon", "Rating"]]
