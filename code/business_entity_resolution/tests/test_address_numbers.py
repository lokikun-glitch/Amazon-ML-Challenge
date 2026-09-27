"""Regression tests for address_numbers.extract (formats taken from the training corpus)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from address_numbers import extract  # noqa: E402


def ex(addr, country):
    return extract(addr, country)


# ----------------------------------------------------------------------------- US

@pytest.mark.parametrize("addr,house,unit", [
    ("1795 Westchester Drive, High Point, NC", "1795", ""),
    ("OH, Columbus, 5559 Orville Avenue", "5559", ""),                       # shuffled components
    ("2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD", "2100", "G"),
    ("5100 Highbridge Street, Unit Unit 15F, Manlius, NY", "5100", "15F"),
    ("Unit APT 605, 2550 Oliver Avenue, Wichita, KS", "2550", "605"),       # unit before street
    ("6203 Winding Creek Way, # 205, Denver, North Carolina", "6203", "205"),  # '# N' segment = unit
    ("#1110 HAWK TRL, COPPERAS COVE, TX", "1110", ""),                       # '#' prefix = house
    ("##16 FAIRLAWN DR, BETHLEHEM, NY", "16", ""),
    ("025238 TANNIN CIRCLE, MILTON, DE", "25238", ""),                       # zero padding stripped
    ("0012019 DEVINE ST, PMB 152, PRESCOTT VALLEY, AZ", "12019", ""),        # PMB is not a unit/house
    ("5527C KINKAID STREET, WICHITA, KS", "5527C", ""),
    ("9207-B Old Mount Vernon Road, FAIRFAX COUNTY, VA", "9207-B", ""),
    ("27W 140 Hickory Lane, West Chicago, IL", "27W", ""),
    ("364 Highway 36, Milano, TX", "364", ""),                                # street-name number ignored
    ("10091 301, Four Oaks, NC", "10091", ""),
    ("TX, Paris, 696 County Road 42560", "696", ""),
    ("5100 Main, Unit C10, Marlborough, MA", "5100", "C10"),
    ("Madison Street, Baltimore, Maryland", "", ""),                          # no number at all
])
def test_us_house_and_unit(addr, house, unit):
    r = ex(addr, "US")
    assert r["primary_house_number"] == house
    assert r["unit_number"] == unit


def test_us_compounds_keep_both_parts():
    r = ex("115-48 197 Street, Saint Albans, NY", "US")
    assert r["primary_house_number"] == "115-48"
    assert r["compound_number"] == "115-48"
    assert r["house_atoms"] == "115|48"
    r = ex("1026 1/2 East Street, Madison, IN", "US")
    assert r["primary_house_number"] == "1026 1/2"
    assert r["house_head"] == "1026"


def test_us_decoy_numbers_differ():
    a = ex("1182 Trapper Loop, Garden City, UT", "US")
    b = ex("1191 TRAPPER LOOP, GARDEN CITY, UT", "US")
    assert a["primary_house_number"] != b["primary_house_number"]
    a = ex("2791 Timpview Drive, Provo, UT", "US")
    b = ex("002798 TIMPVIEW DRIVE, PROVO, UT", "US")
    assert (a["house_head"], b["house_head"]) == ("2791", "2798")
    # zero padding alone must NOT look like a different number
    assert ex("00209 Gold Street, JUNEAU CDP, AK", "US")["primary_house_number"] == \
        ex("209 Gold Street, Juneau, AK", "US")["primary_house_number"]


def test_us_zip_never_read_as_house_number():
    r = ex("1600 Main Street, Springfield, IL 62701", "US")
    assert r["postal_code"] == "62701"
    assert r["primary_house_number"] == "1600"
    assert "62701" not in r["number_tokens"].split()
    r = ex("12 Oak Avenue, Albany, New York 12207-1234", "US")
    assert r["postal_code"] == "12207"


@pytest.mark.parametrize("addr", [
    "17560 Ellis Road, Tahlequah, OK",                  # 5-digit HOUSE number (old postal_code bug)
    "TX, Paris, 696 County Road 42560",                 # trailing road number is not a ZIP
    "Portland, 400 Congress Street, ME, Unit UNIT 15308",
    "36234 Aspen Court, WI, City Of Independence, Fl 13887",   # 'Fl' = floor, not Florida
    "005424 OLDE VINTAGE DR, HILLIARD, OH",
])
def test_us_no_false_zip(addr):
    assert ex(addr, "US")["postal_code"] == ""


def test_us_house_number_not_confused_with_postal():
    r = ex("17560 Ellis Road, Tahlequah, OK", "US")
    assert r["primary_house_number"] == "17560"


# ----------------------------------------------------------------------------- India

@pytest.mark.parametrize("addr,house,compound", [
    ("26/244 Shivaji Nagar, Madanganj, Kishangarh, Ajmer, Rajasthan", "26/244", "26/244"),
    ("D. No: 6-3-668/10/4, Durga Nagar Colony, Punjagutta, Hyderabad, Telangana", "6-3-668/10/4", "6-3-668/10/4"),
    ("B-1/307, 3Rd Floor New Ashok Nagar, Delhi", "B-1/307", "B-1/307"),
    ("H.No. 36/19, Venkat Rao Colony, Secunderabad, Telangana", "36/19", "36/19"),
    ("Plot No 414/A, Sec 68, Imt, Ballabgarh, Faridabad, Haryana", "414/A", ""),
    ("No.261/A, 1St Floor, Bommasandra Industrial Area, Bangalore, Karnataka", "261/A", ""),
    ("43, Shiva Complex, Ved Nagar, Patna, Bihar", "43", ""),
    ("#884/7 Meenal Arcade Nal, Pune City, MH", "884/7", "884/7"),
    ("Door No #8 Yadaval Street, Chennai, TN", "8", ""),
    ("26-5/1A, Mallayyapalem, Madhurawada, Visakhapatnam", "26-5/1A", "26-5/1A"),
    ("E-179, GROUND FLOOR GREATER KAILASH - I, NEW DELHI, Delhi", "E-179", ""),
    ("Mouza - Dahua, Tola- Hechla P S And Anchal - Baunshi No 440, Banka, Bihar", "440", ""),
])
def test_india_house_and_compound(addr, house, compound):
    r = ex(addr, "India")
    assert r["primary_house_number"] == house
    assert r["compound_number"] == compound


def test_india_compound_keeps_second_component():
    a = ex("26/244 Shivaji Nagar, Madanganj, Rajasthan", "India")
    b = ex("26/248 SHIVAJI NAGAR, MADANGANJ, Rajasthan", "India")
    assert a["house_head"] == b["house_head"] == "26"          # old street_number saw only this
    assert a["primary_house_number"] != b["primary_house_number"]
    assert set(a["house_atoms"].split("|")) == {"26", "244"}


def test_india_units_are_not_house_numbers():
    r = ex("Flat No 504, Pno 4, 5, 6, Gothic Pangea, Hyd, Hyderabad, Telangana", "India")
    assert r["unit_number"] == "504"
    assert r["primary_house_number"] == "4"
    r = ex("Pno:10, 11&12, Fno:201, Svr Estates Nagaram, Rangareddy, Telangana", "India")
    assert r["primary_house_number"] == "10"
    assert r["unit_number"] == "201"
    r = ex("Shop No.309, Shubham Tower, Near Neelam Chowk, Faridabad, Haryana", "India")
    assert r["unit_number"] == "309" and r["primary_house_number"] == ""


def test_india_context_numbers_are_not_house_numbers():
    for addr in ["Sector 19, Vashi, Navi Mumbai, Maharashtra",
                 "Ward No. - 07, Near Kumar Hardware, Motihari, Bihar",
                 "Level 5, Max House, Okhla, New Delhi",
                 "12Th Main 2Nd Cross, Raghavendra Block, Bangalore",
                 "National Highway No. 8A, Morbi, Gujarat"]:
        assert ex(addr, "India")["primary_house_number"] == "", addr


def test_india_parcel_is_fallback_only():
    r = ex("Shed No-127, S.No. 43/1, Off Gondal Road Vavdi, Rajkot, Gujarat", "India")
    assert r["primary_house_number"] == "43/1"             # parcel used only because nothing better
    r = ex("Plot No. 3, Kh No. 2132 & 2133 Amar Colony, Nangloi, Delhi", "India")
    assert r["primary_house_number"] == "3"


def test_india_pin():
    r = ex("Abban House, Observatory Post Kodaikanal-624 103, Dindigul, Tamil Nadu", "India")
    assert r["postal_code"] == "624103"
    assert "624" not in r["number_tokens"].split() and "103" not in r["number_tokens"].split()
    assert ex("123 Main Road, Delhi, 110001", "India")["postal_code"] == "110001"
    assert ex("NO. 00955 HOUSE NO. 18/384, MAITHAN, AGRA", "India")["postal_code"] == ""


def test_ordinals_are_not_numbers():
    r = ex("3Rd Main, E Block, 2Nd Stage Rajaji Nagar, Bangalore", "India")
    assert r["number_tokens"] == ""


# ----------------------------------------------------------------------------- ambiguous / degenerate

@pytest.mark.parametrize("addr", ["", None, "<NULL>", "C/O Asok Kumar Datta, Kolkata, WB",
                                  "Vill-Golhnamau, Thana-Sujanganj, Jaunpur, Uttar Pradesh"])
def test_empty_or_numberless(addr):
    r = ex(addr, "India")
    assert r["primary_house_number"] == "" and r["number_tokens"] == ""


def test_unlabelled_mid_segment_number_not_promoted():
    # "Road No 2" is context; the unlabelled "#504" at the END of a segment list is ambiguous
    r = ex("Tamil Nadu, Chennai, No.2 Road, Railady", "India")
    assert r["primary_house_number"] == "2"   # explicit 'No.' label wins (literal reading, no semantics)
    r = ex("Kolkata, Ce-107, Salt Lake City, Sector-1", "India")
    assert r["primary_house_number"] == "CE-107"


def test_deterministic():
    a = "D. No: 6-3-668/10/4, Durga Nagar Colony, Punjagutta, Hyderabad"
    assert ex(a, "India") == ex(a, "India")


# ----------------------------------------------------------------------------- known limitation (not fixed)
# A number glued to the following word ("2/1Chattaarpur", "889Bvettoor") loses its trailing
# component or the whole number. Found in the 310-case forensic check (2 cases); ~1.5% of Indian
# and ~0% of US raw addresses. Deliberately NOT fixed: the A/B/C VAL results were already
# computed with the current extractor and re-running would be a second look at VAL.
@pytest.mark.xfail(strict=True, reason="glued number+word not split; frozen for the A/B/C experiment")
def test_india_number_glued_to_word():
    assert ex("2/1Chattaarpur Mor Mehrauli, New Delhi", "India")["primary_house_number"] == "2/1"
