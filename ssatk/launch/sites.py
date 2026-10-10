# Launch pad information
launch_pads = {
    "Kennedy Space Center LC-39A": {
        "latitude": 28.6082, "longitude": -80.6041,
        "typical_trajectory": {"azimuth_range": "40°-55°",
                               "description": "Northeast over Atlantic for LEO/GTO"}
    },
    "Kennedy Space Center LC-39B": {
        "latitude": 28.6272, "longitude": -80.6209,
        "typical_trajectory": {"azimuth_range": "40°-55°",
                               "description": "Northeast over Atlantic for LEO/GTO"}
    },
    "Cape Canaveral SLC-40": {
        "latitude": 28.5620, "longitude": -80.5772,
        "typical_trajectory": {"azimuth_range": "40°-90°",
                               "description": "Northeast to East for LEO/GTO"}
    },
    "Vandenberg SLC-4E": {
        "latitude": 34.6320, "longitude": -120.6107,
        "typical_trajectory": {"azimuth_range": "180°-210°",
                               "description": "Southward for polar orbits"}
    },
    "Wallops Flight Facility Pad 0A": {
        "latitude": 37.8338, "longitude": -75.4881,
        "typical_trajectory": {"azimuth_range": "90°-135°",
                               "description": "Eastward for LEO/suborbital"}
    },
    "Baikonur Cosmodrome Site 1/5": {
        "latitude": 45.9203, "longitude": 63.3422,
        "typical_trajectory": {"azimuth_range": "50°-65°",
                               "description": "Northeast for LEO"}
    },
    "Plesetsk Cosmodrome Site 43/4": {
        "latitude": 62.9278, "longitude": 40.4572,
        "typical_trajectory": {"azimuth_range": "0°-45°",
                               "description": "Northward for polar orbits"}
    },
    "Vostochny Cosmodrome Site 1S": {
        "latitude": 51.8846, "longitude": 128.3347,
        "typical_trajectory": {"azimuth_range": "50°-65°",
                               "description": "Northeast for LEO"}
    },
    "Jiuquan Launch Center LC-43/91": {
        "latitude": 40.9675, "longitude": 100.2911,
        "typical_trajectory": {"azimuth_range": "0°-45°",
                               "description": "Northward for polar orbits"}
    },
    "Xichang Launch Center LC-2": {
        "latitude": 28.2460, "longitude": 102.0267,
        "typical_trajectory": {"azimuth_range": "90°-120°",
                               "description": "Eastward for GTO/LEO"}
    },
    "Taiyuan Launch Center LC-9": {
        "latitude": 38.8493, "longitude": 111.6081,
        "typical_trajectory": {"azimuth_range": "0°-45°",
                               "description": "Northward for polar orbits"}
    },
    "Satish Dhawan Space Centre SLP": {
        "latitude": 13.7199, "longitude": 80.2304,
        "typical_trajectory": {"azimuth_range": "90°-105°",
                               "description": "Eastward for GTO/polar"}
    },
    "Guiana Space Centre (Kourou) ELA-3": {
        "latitude": 5.2378, "longitude": -52.7753,
        "typical_trajectory": {"azimuth_range": "0°-90°",
                               "description": "North to East for GTO/LEO"}
    },
    "Tanegashima Space Center LP-1": {
        "latitude": 30.4000, "longitude": 130.9750,
        "typical_trajectory": {"azimuth_range": "90°-135°",
                               "description": "Eastward for GTO/LEO"}
    },
    "SpaceX South Texas (Boca Chica)": {
        "latitude": 25.9972, "longitude": -97.1561,
        "typical_trajectory": {"azimuth_range": "90°-135°",
                               "description": "Eastward for LEO/GTO"}
    },
    "Rocket Lab Launch Complex 1 (New Zealand)": {
        "latitude": -39.2620, "longitude": 177.8648,
        "typical_trajectory": {"azimuth_range": "135°-180°",
                               "description": "Southeast for sun-synchronous"}
    }
}

landing_pads = {
    # ───────────────────────── NASA / propulsion ─────────────────────────
    "NASA Stennis Space Center": {
        "latitude": 30.3627667,
        "longitude": -89.6002000,
        "typical_testing": {
            "domain": "Rocket propulsion (engine/stage static fire)",
            "description": "Major U.S. rocket propulsion test complex; used for large liquid engines and stages.",
        },
    },

    # ───────────────────────── Missile / range complexes ─────────────────
    "White Sands Missile Range (NM)": {
        "latitude": 32.3355600,
        "longitude": -106.4058300,
        "typical_testing": {
            "domain": "Missile/rocket flight test, range instrumentation",
            "description": "Large overland test range used for missile and rocket testing in the Southwest.",
        },
    },

    "Utah Test and Training Range (UT)": {
        "latitude": 41.0000000,
        "longitude": -113.2500000,
        "typical_testing": {
            "domain": "Overland weapons test / training range",
            "description": "Desert range supporting weapons testing/training with large restricted airspace/ground areas.",
        },
    },

    "Eglin Air Force Base / Eglin Test Range (FL)": {
        "latitude": 30.4894400,
        "longitude": -86.5422200,
        "typical_testing": {
            "domain": "Air-delivered weapons testing",
            "description": "Weapons development/test activities, with test corridors generally oriented toward the Gulf.",
        },
    },

    "Pacific Missile Range Facility Barking Sands (HI)": {
        "latitude": 22.0227800,
        "longitude": -159.7850000,
        "typical_testing": {
            "domain": "Ocean-based missile/target testing & tracking",
            "description": "Instrumented naval test & training range with large controlled airspace/underwater range.",
        },
    },

    "Naval Base Ventura County – NAS Point Mugu (CA)": {
        "latitude": 34.1192765,
        "longitude": -119.1195889,
        "typical_testing": {
            "domain": "Naval air / weapons testing; sea-range access",
            "description": "Coastal naval air station supporting test activities with Pacific overwater corridors.",
        },
    },

    "Naval Air Weapons Station China Lake (CA)": {
        "latitude": 35.6851667,
        "longitude": -117.6929444,
        "typical_testing": {
            "domain": "Weapons/ordnance test & evaluation (overland desert range)",
            "description": "Large inland naval weapons station used for flight/weapon test and evaluation.",
        },
    },

    "Naval Air Station Patuxent River (MD)": {
        "latitude": 38.2861100,
        "longitude": -76.4116700,
        "typical_testing": {
            "domain": "Naval aviation test & evaluation",
            "description": "Navy flight test center activities centered on the Patuxent River complex.",
        },
    },

    "Aberdeen Proving Ground (MD)": {
        "latitude": 39.4602806,
        "longitude": -76.1250000,
        "typical_testing": {
            "domain": "Army test & evaluation (weapons, survivability, etc.)",
            "description": "Major U.S. Army proving ground for test and evaluation programs.",
        },
    },

    "Dugway Proving Ground (UT)": {
        "latitude": 40.2208300,
        "longitude": -112.7441700,
        "typical_testing": {
            "domain": "CBRN defense / test & evaluation",
            "description": "Army test center historically associated with chemical/biological defense-related testing.",
        },
    },

    # ───────────────────────── Specialized / notable test sites ──────────
    "Arnold Engineering Development Complex (TN)": {
        "latitude": 35.3790000,
        "longitude": -86.0500000,
        "typical_testing": {
            "domain": "Aerospace ground test (wind tunnels / propulsion / aerothermal)",
            "description": "USAF test complex supporting aerospace system ground testing and evaluation.",
        },
    },

    "Holloman High Speed Test Track (NM)": {
        "latitude": 32.8848639,
        "longitude": -106.1499333,
        "typical_testing": {
            "domain": "High-speed sled / track-based test",
            "description": "High-speed ground test track used for sled tests and related instrumentation work.",
        },
    },

    "Tonopah Test Range Airport (NV)": {
        "latitude": 37.7988333,
        "longitude": -116.7801667,
        "typical_testing": {
            "domain": "Range support airfield (test/training operations)",
            "description": "Remote range support airfield associated with test and training activities in Nevada.",
        },
    },

    "Yuma Proving Ground – Laguna Army Airfield (AZ)": {
        # Airfield used as a clean coordinate anchor for the YPG complex
        "latitude": 32.8645806,
        "longitude": -114.3929750,
        "typical_testing": {
            "domain": "Army test & evaluation (desert range)",
            "description": "Army proving ground supporting diverse test & evaluation activities in the Yuma area.",
        },
    },
}
