"""Build the broad 50 cm ape actuator research inventory.

The source register below is intentionally inclusive. Blank specifications mean
unverified, not zero. Run this file after editing the register to refresh both
the engineering CSV and readable Markdown tables.
"""

import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROWS = []
S = {
    "feetech": "https://www.feetechrc.com/sts_ttl_series%20servo.html",
    "feetech_small": "https://www.feetechrc.com/products/sts_ttl_series%20servo-page-2",
    "feetech_all": "https://www.feetechrc.com/products",
    "waveshare": "https://www.waveshare.com/product/st3215-servo.htm",
    "hiwonder": "https://docs.hiwonder.com/en/latest/new-index.html",
    "robotis": "https://emanual.robotis.com/docs/en/dxl/x/",
    "robotis_old": "https://emanual.robotis.com/docs/en/dxl/",
    "cubemars": "https://www.cubemars.com/goods.php?id=1031",
    "robstride": "https://www.robstride.com/",
    "steadywin": "https://www.steadywin.cn/en/pr.jsp?_pp=0_753_3",
    "myactuator": "https://files.seeedstudio.com/products/Myactuator/User_Manual_X_Series_Products_V1.1.pdf",
    "harmonic": "https://www.harmonicdrive.net/products/miniature-products",
    "makon": "https://www.makongear.com/product/humanoid-joint-actuator/",
    "actuonix": "https://www.actuonix.com/assets/images/ACX_Product_Table_MASTER_2025_OUT_10-20%20%281%29.pdf",
}


def add(category, maker, model, roles, source, *, origin="manufacturer", form="complete servo",
        control="", voltage="", mass="", size="", torque="", basis="", force="", price="", note=""):
    ident = re.sub(r"[^a-z0-9]+", "-", f"{maker}-{model}".lower()).strip("-")
    if any(row["id"] == ident for row in ROWS):
        raise ValueError(f"duplicate ID: {ident}")
    ROWS.append(dict(id=ident, category=category, maker=maker, model=model, origin=origin,
                     form=form, possible_roles=roles, control_interface=control, voltage_v=voltage,
                     mass_g=mass, envelope_mm=size, torque_nm=torque, torque_basis=basis,
                     force_n=force, observed_retail_usd=price, observed_price_usd=price,
                     price_basis="observed retail listing" if price != "" else "", source_url=source,
                     engineering_note=note))


def group(category, maker, models, roles, source, **kwargs):
    for model in models.split(";"):
        add(category, maker, model.strip(), roles, source, **kwargs)


# Existing source-measured package references. All torques below retain the
# source's rated, stall, peak or maximum label; no duty curve is inferred.
add("Serial smart servos", "Feetech", "ST-3215-C001", "arm; ankle; small leg",
    "https://files.seeedstudio.com/products/Feetech/108090023_STS3215-C001_Datasheet.pdf",
    control="TTL", voltage="4-7.4", mass=55, size="45.2x24.7x35", torque=.490,
    basis="rated at 7.4 V", note="Separate 7.4 V variant")
add("Serial smart servos", "Feetech", "ST-3215-C018", "arm; ankle; leg comparison",
    "https://www.feetechrc.com/Data/feetechrc/upload/file/20240507/6385067068652648096680943.pdf",
    control="TTL", voltage="4-14", mass=55, size="45.2x24.7x35", torque=.981,
    basis="rated at 12 V", note="Current arm/ankle reference; no proven gait duty")
add("Serial smart servos", "Feetech", "ST-3250-C001", "hip; knee; shoulder",
    "https://www.feetechrc.com/Data/feetechrc/upload/file/20240120/6384135881578380868917773.pdf",
    control="TTL", voltage="6-12", mass=74.5, size="45.22x24.72x35", torque=1.569,
    basis="rated at 12 V", note="Current larger leg reference; regulated rail")
add("Serial smart servos", "Feetech", "ST-3215-C046", "arm; ankle; faster leg comparison",
    "https://files.seeedstudio.com/products/Feetech/101090142_Feetech_ST-3215-C046_Datasheet.pdf",
    control="TTL", voltage="5-8.4", mass=55, size="45.2x24.7x35", torque=.471,
    basis="rated at 7.4 V", note="1:147 gearing; 14.4 kgf cm stall at 7.4 V")
add("PWM / RC servos", "Feetech", "FT-3332-C001", "knee; arm; compact leg",
    "https://www.feetechrc.com/Data/feetechrc/upload/file/20260623/6391782192726766615980211.pdf",
    control="PWM", voltage="8.4 rating", mass=51.5, size="34x15x36.5", torque=1.282,
    basis="rated at 8.4 V", note="Slim case; limited angular travel")
group("Serial smart servos", "Feetech", "ST-3036-C001;ST-3036-C002;ST-3032-C001;ST-3032-C036",
      "head; hand; wrist; elbow", S["feetech_small"], control="TTL", voltage="6 rating",
      torque=.441, basis="stall at 6 V", note="Verify exact suffix dimensions and rated duty")
group("Serial smart servos", "Feetech", "ST-3025-C001;STS3025BL;ST-3046-C001;ST-3235-C001;SM-2930-C001;SM-2930-C002",
      "arm; ankle; leg comparison", S["feetech_all"], control="TTL or RS485; check exact model",
      note="Catalog lead; configure exact voltage, case, torque and bus")

# Waveshare's own comparison table calls these locked-rotor values. Conversion
# uses 1 kgf cm = 0.0980665 Nm; source SKU can differ from Feetech namesakes.
for model, kgcm, volts in [
    ("SC09", 2.3, "4-6"), ("SC15", 15, "4.8-8.4"), ("ST3020", 25, "6-14"),
    ("ST3215 7.4 V", 19.5, "4-7.4"), ("ST3215 12 V", 30, "6-12.6"),
    ("ST3025", 40, "6-12.6"), ("ST3235", 30, "6-12.6"),
    ("CF35-12", 35, "9-12.6"), ("RSBL35-24-HS", 35, "9-24"),
    ("RSBL45-24", 45, "9-24"), ("RSBL85-12", 85, "9-12"),
    ("RSBL85-24", 85, "9-24"), ("RSBL120-24", 120, "9-24")]:
    add("Serial smart servos", "Waveshare", model,
        "head; hand" if kgcm < 10 else ("arm; ankle" if kgcm < 25 else "hip; knee; shoulder"),
        S["waveshare"], origin="seller", control="RS485" if model.startswith("RSBL") else "TTL",
        voltage=volts, torque=round(kgcm * .0980665, 3), basis="advertised locked-rotor",
        note="Seller SKU; confirm OEM and sustained output")

add("Serial smart servos", "Hiwonder", "HX-35H", "hip; knee; shoulder",
    "https://www.hiwonder.com/products/hx-35h", control="UART bus", voltage="9-12.6",
    mass=52, size="45.2x24.7x35", torque=3.432, basis="static maximum", price=18.99,
    note="240-degree travel; 25 kgf cm rotation claim is separate")
add("Serial smart servos", "Hiwonder", "HX-12H", "arm; ankle; head",
    "https://www.hiwonder.com/products/hx-12h", control="UART bus", torque=1.177,
    basis="advertised torque; duty unstated", price=16.99, note="Check exact plug and travel")
group("Serial smart servos", "Hiwonder",
      "HX-06L;HX-20L;HX-35HM;HX-65HM;HTD-30H;HTD-35H;HTD-45H;HTS-16L;HTS-20L;HTS-20H;HTS-21H;HTS-30HS;HTS-35H;LX-15D;LX-16A;LX-224;LX-824;LX-224HV;LX-824HV;LX-1501;LX-225",
      "arm; ankle; head; gripper; leg comparison", S["hiwonder"],
      control="Hiwonder bus; verify protocol and rail", note="Indexed model lead; exact drawing and duty pending")

# Official X-series table. 12 V and 5 V values are stall values, per ROBOTIS.
for model, torque, volts, roles in [
    ("XL330-M077-T", .215, 5, "head; hand"), ("XL330-M288-T", .52, 5, "head; hand; wrist"),
    ("XC330-M181-T", .60, 5, "head; wrist"), ("XC330-M288-T", .93, 5, "head; elbow"),
    ("XC330-T181-T", .80, 12, "head; elbow"), ("XC330-T288-T", 1.00, 12, "head; elbow; arm"),
    ("XM335-T323-T", 1.12, 12, "arm; ankle"), ("XL430-W250-T", 1.5, 12, "arm; ankle"),
    ("2XL430-W250-T", 1.5, 12, "paired shoulder; paired hip"),
    ("XC430-W150-T", 1.6, 12, "arm; ankle"), ("XC430-W240-T", 1.9, 12, "arm; knee"),
    ("XC430-T150BB-T", 1.6, 12, "arm; ankle"), ("XC430-T240BB-T", 1.9, 12, "arm; knee"),
    ("2XC430-W250-T", 1.8, 12, "paired hip; paired shoulder"),
    ("XM430-W210-T", 3.0, 12, "hip; knee"), ("XM430-W350-T", 4.1, 12, "hip; knee"),
    ("XH430-W210-T", 2.5, 12, "hip; knee"), ("XH430-W350-T", 3.4, 12, "hip; knee"),
    ("XH430-V210-R", 2.6, 24, "hip; knee"), ("XH430-V350-R", 3.3, 24, "hip; knee")]:
    add("DYNAMIXEL servos", "ROBOTIS", model, roles, S["robotis"],
        control="DYNAMIXEL 2.0; suffix determines TTL/RS485", voltage=f"{volts} table point",
        torque=torque, basis="stall at table voltage", note="Exact e-manual needed for package, current and thermal curve")
group("DYNAMIXEL servos", "ROBOTIS", "AX-12A;AX-18A;MX-12W;MX-28T;MX-64T;MX-106T",
      "aftermarket arm; ankle; head; leg comparison", S["robotis_old"], origin="aftermarket / legacy",
      control="DYNAMIXEL; protocol varies", note="Used/legacy lead; inspect wear, firmware and price")

# Other serial ecosystems matter because the body need not use a single vendor.
for model, static_kgcm, dynamic_kgcm in [("LSS-HS1", 14, 2.8),
                                          ("LSS-ST1", 8, 1.6),
                                          ("LSS-HT1", 29, 5.8)]:
    add("Serial smart servos", "Lynxmotion", model, "arm; ankle; head; leg comparison",
        "https://wiki.lynxmotion.com/info/wiki/lynxmotion/view/ses-v2/lynxmotion-smart-servo/lss-specifications/",
        control="LSS serial", voltage="6-12.6", torque=round(dynamic_kgcm * .0980665, 3),
        basis="published dynamic ~25% of static",
        note=f"Static maximum {static_kgcm} kgf cm; vendor warns thermal shutdown around 10 min at 25% load")
add("Serial smart servos", "Kondo", "KRS-3304R2 ICS", "arm; ankle; head; compact leg",
    "https://kondo-robot.com/product/krs-3304r2-ics", control="ICS 3.6 serial/PWM",
    voltage="6-7.4", mass=33.7, size="32.5x26x26", torque=1.363,
    basis="maximum at 7.4 V", note="270-degree travel; current-position readback supported")
group("Serial smart servos", "Kondo", "KRS-3301 ICS;KRS-3302 ICS;KRS-2552R2HV ICS",
      "arm; ankle; head; compact leg",
      "https://kondo-robot.com/w/wp-content/uploads/KRS-series_manual_Download_a4-En-V2_1.pdf",
      control="ICS serial/PWM", note="Configure exact version, current readback and voltage")

add("PWM / RC servos", "DSservo", "DS3218", "arm; ankle; leg coupon",
    "https://www.dsservo.com/d_file/DS3218%20datasheet.pdf", control="PWM", mass=60,
    size="40x20x40.5", torque=1.961, basis="advertised stall", note="No joint-state bus")
add("PWM / RC servos", "DSservo", "DS3225", "arm; ankle; leg coupon",
    "https://www.dsservo.com/d_file/DS3225%20datasheet.pdf", control="PWM", mass=60,
    size="40x20x40.5", torque=2.452, basis="advertised stall", note="No joint-state bus")
add("PWM / RC servos", "AGFRC", "A80BHS V2", "hip; knee; shoulder",
    "https://www.agf-rc.com/agfrc-a80bhs-v2-36kg-high-speed-brushless-servo-p1637348.html",
    control="PWM", voltage="4.8-8.4", mass=80, size="40x20x37.5", torque=3.530,
    basis="stall at 8.4 V", price=79.99, note="No joint-state bus; 180-degree travel")
add("PWM / RC servos", "Hitec", "D955TW", "hip; knee; shoulder",
    "https://hitecrcd.com/d955tw-32-bit-high-torque-titanium-gear-servo/", control="PWM",
    torque=2.844, basis="published torque at 7.4 V", note="Add output sensing")
group("PWM / RC servos", "Hitec", "HS-645MG;HS-7955TG;D645MW;HS-5085MG",
      "arm; gripper; head; leg coupon", "https://hitecrcd.com/servo-specification-chart/",
      control="PWM", note="Compare exact torque/travel and add output sensing")
add("PWM / RC servos", "Savox", "SC1256TG-BE", "arm; ankle; leg coupon",
    "https://www.savoxusa.com/collections/stadium-truck",
    origin="seller", control="PWM", voltage="4.8-6", mass=52.4, size="40.3x20.2x37.2",
    torque=1.961, basis="published torque at 6 V", price=93.49,
    note="Sold-out listing; approximately 130-degree maximum travel; add output sensing")
add("PWM / RC servos", "Savox", "SB-2290SG", "arm; ankle; leg coupon",
    "https://www.savoxtech.com.cn/", control="PWM", note="Manufacturer index lead; exact SKU and output sensing")
group("PWM / RC servos", "DSservo", "DS3115;RDS3115;RDS3120",
      "arm; gripper; leg coupon", "https://www.dsservo.com/en/download.asp",
      control="PWM", note="Original maker download index; select exact data sheet")
group("PWM / RC servos", "KST", "X08H Plus V6.0;X10-710 V8.0;X10 V8.0;X15-908;X20-1035",
      "hand; head; arm; ankle coupon", "https://kstservos.com/collections/x-series",
      control="PWM", note="Thin-wing and standard cases; inspect exact travel and output support")

for model, source, note in [
    ("ST3215 private-label", S["waveshare"], "Verify OEM, voltage variant, firmware and spline"),
    ("LX-16A private-label", S["hiwonder"], "Verify protocol, registers, gear train and source"),
    ("MG996R generic clone", "https://www.towerpro.com.tw/product/mg996r/", "Original maker is reference; do not transfer specs to anonymous clone"),
    ("DS3218 reseller clone", "https://www.dsservo.com/d_file/DS3218%20datasheet.pdf", "Compare sample against original drawings before using specs")]:
    add("Clone / aftermarket leads", "Multi-source", model, "arm; gripper; bench comparison",
        source, origin="clone / aftermarket", form="source class; exact SKU TBD",
        control="varies", note=note)
add("Clone / aftermarket leads", "Spotpear", "ST3215-HS", "arm; ankle; faster joint comparison",
    "https://spotpear.com/wiki/Serial-UART-Bus-Magnetic-Encoder-ST3215-HS-Servo-Motor-20kg-High-Speed.html",
    origin="seller", control="TTL", note="Seller high-speed variant; verify exact OEM, gear ratio and thermal curve")
add("Clone / aftermarket leads", "Spotpear", "ST3215 12 V", "arm; ankle; leg comparison",
    "https://spotpear.com/shop/Serial-UART-Bus-Magnetic-Encoder-ST3215-Servo-30kg-7.4V-19.5kg/ST3215-Servo.html",
    origin="seller", control="TTL", note="Distributor route; confirm exact underlying model and revision")

for model, torque, mass, size in [
    ("AK40-10", 1.3, 185, "D53x37"), ("AK40-10 V3.0", 1.3, 190, "D53x40.2"),
    ("AK45-10", 2.1, 260, "D53x43"), ("AK45-36", 8, 340, "D55x54"),
    ("AK60-6 V3.0", "", "", "")]:
    add("Integrated brushless joints", "CubeMars", model, "hip; knee; shoulder", S["cubemars"],
        form="geared BLDC joint", control="CAN; driver variant", mass=mass, size=size,
        torque=torque, basis="rated" if torque != "" else "",
        note="Check driver inclusion, heat path and mass budget")
for model, peak, mass, size, source in [
    ("RobStride 05", 5.5, 191, "46x46x44", "https://robstride.com/products/robStride05"),
    ("EduLite 05", 6, 242, "46x46x44", "https://www.robstride.com/products/eduLite05"),
    ("RobStride 00", 14, 310, "57x57x51", "https://robstride.com/products/robStride00"),
    ("RobStride 01", 17, 380, "78.5x78.5x40", "https://robstride.com/products/robStride01"),
    ("RobStride 02", 17, 405, "78.5x78.5x45.5", "https://robstride.com/products/robStride02")]:
    add("Integrated brushless joints", "RobStride", model, "hip; knee; shoulder", source,
        form="QDD BLDC joint", control="CAN", mass=mass, size=size, torque=peak,
        basis="advertised peak", note="Find rated point, exact weight and cooling condition")
group("Integrated brushless joints", "SteadyWin",
      "GIM3505-8;GIM3505-9;GIM3505-36;GIM3510-8;GIM4305-10;GIM4310-10;GIM4310-36;GIM4315-8;GIM6010-6;GIM6010-8",
      "hip; knee; shoulder; arm", S["steadywin"], form="planetary BLDC joint",
      control="CAN/RS485; driver variant", note="Selection-table lead; configure housing, driver and torque")
group("Integrated brushless joints", "MyActuator", "RMD-X2-P28;RMD-X4-P12.5;RMD-X4-P24;RMD-X6-P15;RMD-X6-P36",
      "hip; knee; shoulder", S["myactuator"], form="planetary BLDC joint",
      control="CAN or EtherCAT variant", note="Specify exact torque suffix, revision and output bearing")
group("Integrated brushless joints", "DAMIAO", "DM-J4310-2EC V1.1;DM-J4340-2EC;DM-J4340P-2EC;DM-J8009P-2EC",
      "hip; knee; shoulder", "https://damiao.enactic.ai/en/", form="geared BLDC joint",
      control="CAN; check exact revision", note="Manufacturer manuals; configure package, bus, rated duty and price")
add("Integrated brushless joints", "Xiaomi", "CyberGear Micromotor", "hip; knee; shoulder",
    "https://aifitlab.com/products/xiaomi-cybergear-micromotor", origin="seller",
    form="QDD BLDC joint", control="CAN", voltage="16-28", mass=317,
    size="D80.5x36.5", torque=4, basis="rated", note="12 Nm peak; heavy for repeated joints")
add("Integrated brushless joints", "Motionlayer", "Axion R1", "shoulder; hip; knee comparison",
    "https://motionlayer.company/en-us/products/axion-r1-actuator", form="QDD BLDC joint",
    control="CAN/UART/SPI", mass=160, size="D53x26", torque=1,
    basis="claimed continuous", note="3 Nm peak; check thermal mounting")
add("Integrated brushless joints", "Makon", "MA-M30-40-HOL-2-XX", "shoulder; knee; hip comparison",
    S["makon"], form="harmonic integrated joint", control="CAN variant", mass=190,
    size="D40x60", torque=1.8, basis="rated 51:1 option", note="101:1 option 2.4 Nm rated")
add("Integrated brushless joints", "Makon", "MA-M40-52-HOL-2-XX", "hip; knee borderline",
    S["makon"], form="harmonic integrated joint", control="CAN variant", mass=320,
    size="D52x62", torque=3.5, basis="rated 51:1 option", note="101:1 option 5 Nm rated")
for model, torque, mass, size in [("RSF-5B IDT", .44, 77, "D25x35.8"),
                                  ("RSA-8A IDT", 2, 190, "D40x56.5"),
                                  ("FHA-8C Mini", "", 400, "66x50x49")]:
    add("Integrated brushless joints", "Harmonic Drive", model, "head; arm; hip comparison",
        S["harmonic"], form="harmonic servo joint", control="CANopen or external drive; variant",
        mass=mass, size=size, torque=torque,
        basis="allowable continuous" if torque != "" else "",
        note="Check price and catalog heat sink condition")

# Motors are deliberately recorded separately from complete actuators.
group("Bare motors / custom drive", "CubeMars",
      "GL30 KV290;GL35 KV100;GL40 KV70;GL40II KV82.5;GL60 KV25;GL60II KV28;RI20 KV600;RI30 KV260;RI50 KV100;RI60 KV120;RO40 KV140 Lite;RO50 KV108 Lite",
      "head; arm; hip/knee with reduction", "https://store.cubemars.com/",
      form="bare BLDC motor", control="external FOC + encoder",
      note="Add reducer, output bearings, encoder, controller and heat path")
group("Bare motors / custom drive", "T-Motor", "GB2208;GB4106;GB36-1;GB36-2;GB54-1;GB54-2",
      "head; arm; hip/knee with reduction", "https://store.tmotor.com/de/categorys/gimbal-motor",
      form="bare gimbal BLDC", control="external FOC + encoder",
      note="Direct torque may be low; add reduction and output support")
group("Bare motors / custom drive", "iPower", "GM2804;GM3506;GM4108H-120T;GM5208-12;GM5208-24;GM6208-150T",
      "head; arm; hip/knee with reduction", "https://shop.iflight.com/motion-control-systems-cat34/gimbal-motors-cat44",
      form="bare gimbal BLDC", control="external FOC + encoder",
      note="Encoder option varies; add gearing, output support, controller")
for model, source in [
    ("20D metal gearmotor", "https://www.pololu.com/file/0J1688/pololu-20d-metal-gearmotors.pdf"),
    ("25D metal gearmotor", "https://www.pololu.com/category/115/25d-metal-gearmotors"),
    ("37D metal gearmotor", "https://www.pololu.com/file/0J1706/pololu-37d-metal-gearmotors.pdf")]:
    add("Bare motors / custom drive", "Pololu", model, "wheel; gripper; remote cable drive",
        source, form="brushed gearmotor family", control="H-bridge; optional encoder",
        note="Select exact ratio/winding; inspect shaft load and backdrivability")
for maker, model, source in [
    ("Maxon", "ECX FLAT 32 + GPX 32", "https://www.maxongroup.com/medias/sys_master/root/9406758879262/Cataloge-Page-EN-399.pdf"),
    ("Maxon", "ECX FLAT 22 + GPX 22", "https://www.maxongroup.com/medias/sys_master/root/9406757797918/Cataloge-Page-EN-390.pdf"),
    ("FAULHABER", "BXT 3216 + GPT 32", "https://www.faulhaber.com/fileadmin/Import/Media/EN_32GPT_HT_FCH.pdf")]:
    add("Bare motors / custom drive", maker, model, "arm; hip; knee; precision custom joint",
        source, form="configurable motor + gearbox", control="external drive + encoder",
        note="Get exact complete assembly, torque curve and quote")

for model, source, note in [
    ("Berkeley Lite 5010 cycloidal", "https://github.com/HybridRobotics/berkeley-humanoid-lite", "Printed gearbox with source BOM; inspect license and heat"),
    ("Berkeley Lite 6512 cycloidal", "https://github.com/HybridRobotics/berkeley-humanoid-lite", "Larger printed gearbox; proximal comparison"),
    ("ODRI Solo actuator v1", "https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware/blob/master/mechanics/actuator_module_v1/README.md", "Outrunner, optical encoder, two-stage timing belt and printed parts"),
    ("OpenTorque printed planetary", "https://github.com/G-Levine/OpenTorque-Actuator", "STEP, STL, BOM and print instructions available"),
    ("MIT low-cost modular actuator", "https://dspace.mit.edu/server/api/core/bitstreams/f3404f79-1cb4-4e61-a51b-1f4ef8bb4e45/content", "Research design; check CAD release and scale"),
    ("Printed QDD actuator pair", "https://arxiv.org/abs/2202.12395", "Paper reports two designs under $200 each at publication"),
    ("JeongSeoJin QDD actuator", "https://github.com/JeongSeoJin/Quasi-Direct-Drive-Actuator", "Open development design; assess maturity and BOM"),
    ("SimpleFOC gimbal-belt joint", "https://github.com/simplefoc/Arduino-FOC-balancer", "Control reference; create and qualify ape transmission"),
    ("moteus + custom BLDC joint", "https://github.com/mjbots/moteus", "Open controller route; add motor, encoder, gear and support"),
    ("ODrive + custom BLDC joint", "https://docs.odriverobotics.com/v/latest/hardware.html", "Open controller route; add motor, encoder, gear and braking")]:
    add("Open / DIY actuator designs", "Open hardware", model,
        "joint coupon; hip; knee; shoulder; arm", source, origin="open design",
        form="build route", control="project/custom FOC", note=note)

for model, force in [("L8", 75), ("P8", 155), ("PQ12", 50), ("L12", 80),
                     ("L16", 200), ("P16", 300), ("T16", 300), ("S20", 13)]:
    add("Linear / tendon options", "Actuonix", model, "gripper; neck; remote joint linkage",
        S["actuonix"], form="linear actuator", control="RC/feedback/stepper variant",
        force=force, note="Maximum family force; stroke, gearing and duty depend on exact option")


def append_later_sourced_leads():
    """Add the round and open-build identities found after the base search."""
    study = ROOT / "studies/cove50-actuators"

    def read(path):
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    identities = read(study / "gallery/coverage.csv")
    round_rows = {r["catalog_id"]: r for r in read(study / "source-geometry/round/geometry-evidence.csv")}
    custom_rows = {r["catalog_id"]: r for r in read(study / "source-geometry/custom/joint-geometry-ledger.csv")}
    round_prices = {r["catalog_id"]: r for r in read(study / "commercial-round/integrated-round-actuators.csv")}
    open_prices = {r["catalog_id"]: r for r in read(study / "open-builds/routes.csv")}
    original_ids = {r["id"] for r in ROWS}
    # The gallery is generated from this catalog and can already contain a
    # newer source wave on a rebuild. Only the legacy supplement registers
    # belong to this stage; reviewed waves are appended below.
    later = [r for r in identities if r["catalog_id"] not in original_ids
             and r["catalog_id"] in (round_rows.keys() | custom_rows.keys())]
    if len({r["catalog_id"] for r in identities}) != len(identities):
        raise ValueError("Gallery actuator identities must be unique")
    for identity in later:
        ident = identity["catalog_id"]
        round_part = round_rows.get(ident)
        custom_part = custom_rows.get(ident)
        if bool(round_part) == bool(custom_part):
            raise ValueError(f"Later identity needs exactly one source register: {ident}")
        if round_part:
            diameter, axial = round_part["case_diameter_mm"], round_part["case_axial_mm"]
            envelope = f"D{diameter}x{axial}" if diameter and axial else ""
            source = round_part["product_source_url"]
            note = round_part["unresolved_supplier_question"]
            mass = round_part["mass_g"]
            torque = round_part["rated_torque_nm"]
            basis = round_part["rated_point_conditions"] if torque else ""
            checked_on = round_part["source_checked_on"]
            origin = "manufacturer"
            price = round_prices[ident]["observed_price_usd"]
            price_basis = "maker page listed price" if price else ""
            voltage = round_prices[ident]["nominal_voltage_v"]
        else:
            envelope = mass = torque = basis = ""
            source = custom_part["geometry_source_url"]
            note = custom_part["missing_for_complete_joint"]
            checked_on = custom_part["checked_on"]
            origin = "original open project"
            price = open_prices[ident]["observed_price_usd"]
            price_basis = open_prices[ident]["price_basis"] if price else ""
            voltage = ""
        ROWS.append(dict(
            id=ident, category=identity["category"], maker=identity["maker"],
            model=identity["model"], origin=origin, form=identity["form"],
            possible_roles=identity["roles"].replace(";", "; "), control_interface="",
            voltage_v=voltage, mass_g=mass, envelope_mm=envelope, torque_nm=torque,
            torque_basis=basis, force_n="",
            observed_retail_usd=price if round_part else "",
            observed_price_usd=price, price_basis=price_basis,
            source_url=source, source_checked_on=checked_on,
            engineering_note=note))


append_later_sourced_leads()


NEW_SOURCE = ROOT / "studies/cove50-actuators/source-geometry"


def source_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def add_reviewed_round_waves():
    """Keep reviewed identities and claim scopes separate from the older leads."""
    first = source_rows(NEW_SOURCE / "new-round-candidates/candidates.csv")
    second = source_rows(NEW_SOURCE / "round-supplier-sweep-2/candidates.csv")
    recovery = source_rows(NEW_SOURCE / "round-supplier-sweep-2/existing-size-recovery.csv")
    assert len(first) == 18 and len(second) == 37 and len(recovery) == 10
    known = {r["id"] for r in ROWS}
    assert not known.intersection({r["id"] for r in first + second})
    assert len({r["id"] for r in first + second}) == 55
    for r in recovery:
        part = next(p for p in ROWS if p["id"] == r["existing_id"])
        if r["recovery_status"].startswith("sourced"):
            assert r["outer_diameter_mm"] and r["axial_length_mm"]
            part["envelope_mm"] = f'D{r["outer_diameter_mm"]}x{r["axial_length_mm"]}'
            part["source_url"] = r["source_url"]
            part["source_checked_on"] = r["checked_on"]
            part["assembly_scope"] = r["source_model_scope"]
            part["engineering_note"] += f'; revision-scoped case: {r["raw_source_geometry"]}; {r["remaining_gap"]}'
        else:
            assert not r["outer_diameter_mm"] and not r["axial_length_mm"]
            part["engineering_note"] += f'; geometry still unresolved: {r["remaining_gap"]}'

    for r in first + second:
        first_wave = "published_torque_text" in r
        ident = r["id"]
        scope = r["assembly_scope"]
        component = (ident.startswith("robotis-ym") and ("-m001-" in ident or "-b001-" in ident)) or ident.startswith("synapticon-")
        family = ident.startswith("vaxor-") or ident.startswith("open-compact-")
        open_build = ident.startswith("open-")
        component = component or ident.startswith("faulhaber-")
        if component:
            category, form = "Motor cores / drive components", "component route; transmission or motor still required"
        elif open_build:
            category, form = "Open / DIY actuator designs", "build route"
        elif family:
            category, form = "Configurable round joint families", "configurable joint family"
        else:
            category, form = "Integrated brushless joints", "integrated round joint"
        diameter, axial = r["outer_diameter_mm"], r["axial_length_mm"]
        dimension = f"D{diameter}x{axial}" if diameter and axial else ""
        box_dimension = {"open-tiptapmotor-v1": "47x47x47",
                         "makiina-medium-38": "50.1x50.1x47"}.get(ident, "")
        if box_dimension:
            assert box_dimension.replace("x", "") and r["other_envelope_text"]
            dimension = box_dimension
        # A component or a configurable family is not an exact complete-joint
        # envelope. Its published size is preserved in source_dimension_mm.
        case = dimension if not component and not family else ""
        price_raw = r["price_usd"] if first_wave else r["observed_price"]
        currency = "USD" if first_wave else r["price_currency"]
        price = price_raw if currency == "USD" and re.fullmatch(r"\d+(?:\.\d+)?", price_raw) else ""
        price_basis = r["price_basis"]
        torque = r["rated_torque_nm"] if first_wave else r["rated_or_continuous_nm"]
        torque_basis = r["torque_basis"] if first_wave else r["torque_evidence"]
        if not torque or (not first_wave and "rated" not in torque_basis.lower() and "continuous" not in torque_basis.lower() and "nominal indefinitely" not in torque_basis.lower()):
            torque = ""
        source = r["source_url"]
        note = (r["integration_downside"] if first_wave else r["functional_downside"])
        conflict = r["conflicting_published_values"] if first_wave else r["source_gap_or_conflict"]
        note = "; ".join(x for x in (scope, note, conflict) if x)
        control = r["control_interface"] if first_wave else r["driver_and_control"]
        voltage = r["voltage_v"] if first_wave else ""
        if ident == "synapticon-circulo-7-drive-route":
            control = "EtherCAT drive component; input supply 14–58 V continuous (24–48 V nominal), 60 V peak"
            voltage = "14–58 continuous; 60 peak"
            note += "; official drive spec: 18 A maximum input, 8 A_rms continuous / 24 A_rms peak phase; no motor or reducer included"
        output = r["output_interface_evidence"] if first_wave else r["output_topology"]
        mass = r["mass_g"]
        ROWS.append(dict(id=ident, category=category, maker=r["maker"], model=r["model"] if first_wave else r["model_or_route"],
                         origin=r["identity_tier"], form=form, possible_roles="hip; knee; shoulder; ankle; arm; design route",
                         control_interface=control, voltage_v=voltage, mass_g=mass, envelope_mm=case,
                         torque_nm=torque, torque_basis=torque_basis, force_n="",
                         observed_retail_usd=price if not open_build and not component and "historical" not in price_basis else "",
                         observed_price_usd=price, price_basis=price_basis, source_url=source,
                         source_checked_on=r["checked_on"] if first_wave else r["checked_on"],
                         engineering_note=note, assembly_scope=scope, output_topology=output,
                         shell_opportunity=r["aesthetic_opportunity"] if first_wave else r["shell_opportunity"],
                         functional_downside=note,
                         source_dimension_mm=dimension,
                         price_observation_text=f"{price_raw} {currency}".strip() if price_raw.strip() else "",
                         source_wave="round-first" if first_wave else "round-second"))


add_reviewed_round_waves()


def write():
    price_path = ROOT / "studies/cove50-actuators/source-geometry/catalog-price-refresh/observations.csv"
    with price_path.open(newline="") as stream:
        prices = list(csv.DictReader(stream))
    catalog_ids = [row["id"] for row in ROWS]
    if len(catalog_ids) != 267 or len(set(catalog_ids)) != 267:
        raise ValueError("expected 267 unique canonical actuator IDs")
    if [record["catalog_id"] for record in prices] != catalog_ids:
        raise ValueError("price ledger IDs/order must exactly match canonical catalog")
    price_columns = [
        "price_status", "price_checked_on", "price_currency", "price_amount_min",
        "price_amount_max", "price_unit", "price_assembly_scope", "price_configuration",
        "price_source_url", "price_source_type", "price_capture_url", "price_source_sha256",
        "price_confidence", "price_region_tax_shipping", "price_evidence_note",
    ]
    source_columns = [
        "price_status", "checked_on", "currency", "amount_min", "amount_max",
        "price_unit", "assembly_scope", "configuration_or_revision", "price_source_url",
        "source_type", "source_capture_url", "source_sha256", "confidence",
        "region_tax_shipping", "evidence_note",
    ]
    for row, evidence in zip(ROWS, prices):
        for target, source in zip(price_columns, source_columns):
            row[target] = evidence[source]
        if evidence["price_status"] == "public_exact_price":
            amount = evidence["amount_min"]
            if amount != evidence["amount_max"] or not amount:
                raise ValueError(f"exact price must be one amount: {row['id']}")
            if evidence["currency"] == "USD":
                row["observed_retail_usd"] = amount
                row["observed_price_usd"] = amount
            row["price_basis"] = "exact public unit listing; see price scope/date/source columns"
            row["price_observation_text"] = f"{amount} {evidence['currency']} exact listed unit"
        elif evidence["price_status"] == "public_configuration_range_not_joint_price":
            row["price_observation_text"] = (
                f"{evidence['amount_min']}-{evidence['amount_max']} {evidence['currency']} "
                "reviewed configuration options; no selected joint price"
            )
    for row in ROWS:
        row.setdefault("source_checked_on", "2026-10-07")
        row["baseline_qty"] = {
            "feetech-st-3250-c001": 8,
            "feetech-st-3215-c018": 10,
            "robotis-xl330-m288-t": 3,
        }.get(row["id"], 0)
        row["fit_status"] = ("provisional frame reference; physical duty unqualified"
                             if row["baseline_qty"] else "not screened on 50 cm frame")
    cols = ["id", "category", "maker", "model", "origin", "form", "possible_roles",
            "control_interface", "voltage_v", "mass_g", "envelope_mm", "torque_nm",
            "torque_basis", "force_n", "observed_retail_usd", "observed_price_usd",
            "price_basis", "source_url", "source_checked_on",
            "baseline_qty", "fit_status", "engineering_note", "assembly_scope",
            "output_topology", "source_dimension_mm", "price_observation_text", "source_wave",
            "shell_opportunity", "functional_downside"] + price_columns
    for row in ROWS:
        for col in cols:
            row.setdefault(col, "")
    csv_path = ROOT / "robot/ape-actuator-parts.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(ROWS)

    counts = Counter(row["category"] for row in ROWS)
    lines = [
        "# 50 cm ape actuator landscape", "",
        f"**Research inventory, updated 8 October 2026. {len(ROWS)} distinct model, family, and build-route leads.** "
        "[Machine-readable parts list](../../robot/ape-actuator-parts.csv). This is a broad search register, "
        "not a selected or physically qualified BOM. The current 21-axis frame uses 8 ST-3250-C001, "
        "10 ST-3215-C018 and 3 XL330-M288-T as provisional references; "
        "[frame contract](../../experiments/compact_morphology/ape_frame50/frame_contract.yaml) and "
        "[drive budget](../design/ape-concepts/50cm-drive-duty-axis-budget.md) define the context.", "",
        "Each CSV row has a stable ID, origin, possible roles, voltage, mass, case size, torque basis, force, "
        "observed price, price basis and source. Empty cells mean **not verified**. Family rows are leads to configure, "
        "not purchasable SKUs. Rated, stall, peak and maximum claims retain their original meaning; none implies "
        "sustained gait duty. Linear drive force is in `force_n`, separate from rotary torque in `torque_nm`. "
        "The [267-ID price evidence ledger](../../studies/cove50-actuators/source-geometry/catalog-price-refresh/observations.csv) "
        "separates exact public prices, maker open/quote pricing, selectable component ranges, historical build costs, "
        "unmatched revisions, and no reliably verified offer. The [method and limitations](../../studies/cove50-actuators/source-geometry/catalog-price-refresh/README.md) "
        "explain source and assembly scope. The price checked-on field is the status-audit date; "
        "older un-rechecked observations are not recertified by it. A bare motor or BOM estimate is not the price of an installed joint; "
        "none of these values is a production or Swiss landed quotation.", "",
        "To compare a shortlist, obtain exact revision/drawing, complete driver and encoder, output bearing "
        "and retention, ratio, cable bend, loaded torque-speed curve, thermal mounting and quantity price. "
        "A 500 g joint may be useful as one proximal coupon but cannot populate a 3.2 kg, 21-axis body. "
        "PWM servos need independent joint-state sensing for the planned control experiments.", "",
        "The [first reviewed cylindrical wave](../../studies/cove50-actuators/source-geometry/new-round-candidates/README.md) adds 18 leads; "
        "the [second manufacturer/open wave](../../studies/cove50-actuators/source-geometry/round-supplier-sweep-2/README.md) adds 37. "
        "Six earlier SteadyWin/MyActuator IDs gained revision-scoped D×L; four remain unsized in the "
        "[recovery ledger](../../studies/cove50-actuators/source-geometry/round-supplier-sweep-2/existing-size-recovery.csv). "
        "Motor cores, annular drives, configurable families, and open build frameworks are cataloged as routes, "
        "not as complete actuator cases. A published D×L for such a component remains in `source_dimension_mm` and is "
        "excluded from the selected-shell case-contact sweep. The three Avatar/Atar TD variants and PlaPivot PP-14/17/20 "
        "retain unresolved published torque-label or configuration conflicts; no continuous claim is inferred.", "",
        "## Search map", "", "| Category | Leads |", "|---|---:|",
    ]
    lines += [f"| {category} | {count} |" for category, count in counts.items()]
    lines += ["", "## Candidate register", "",
              "Possible roles are search suggestions, not fit or performance approval. `D` in a size means diameter.", ""]
    for category in counts:
        lines += [f"### {category}", "",
                  "| Model / source | Possible roles | Output claim | Mass | Envelope | Observed price | Shell opportunity | Engineering note |",
                  "|---|---|---|---:|---|---|---|---|"]
        for row in ROWS:
            if row["category"] != category:
                continue
            claim = (f"{row['torque_nm']} N·m {row['torque_basis']}" if row["torque_nm"] != ""
                     else (f"{row['force_n']} N max family force" if row["force_n"] != "" else "—"))
            mass = f"{row['mass_g']} g" if row["mass_g"] != "" else "—"
            status = row["price_status"]
            if status == "public_exact_price":
                price_label = f"{row['price_amount_min']} {row['price_currency']} exact {row['price_assembly_scope']}"
            elif status == "public_configuration_range_not_joint_price":
                price_label = f"{row['price_amount_min']}–{row['price_amount_max']} {row['price_currency']} options; no selected joint"
            elif status == "prior_public_price_not_rechecked":
                price_label = f"Prior {row['price_amount_min']} {row['price_currency']}; not rechecked"
            elif status == "historical_builder_cost":
                price_label = f"Historical builder/BOM {row['price_amount_min']} {row['price_currency']}"
            else:
                price_label = status.replace('_', ' ')
            price_text = (f"[{price_label}]({row['price_source_url']})"
                          if row['price_source_url'] else price_label)
            lines.append(f"| [{row['maker']} {row['model']}]({row['source_url']}) | "
                         f"{row['possible_roles']} | {claim} | {mass} | {row['envelope_mm'] or '—'} | "
                         f"{price_text.replace('|', '/')} | {row.get('shell_opportunity', '').replace('|', '/') or '—'} | "
                         f"{row['engineering_note'].replace('|', '/')} |")
        lines.append("")
    lines += ["## How to use this register", "",
              "- Manufacturer and original-project sources support a published specification or design lead, "
              "subject to exact revision. Seller links support a listing, not equivalence to a similar-looking OEM part. "
              "Clone and aftermarket entries require incoming inspection before applying original specifications.",
              "- Blank mass, voltage and torque fields are next purchasing questions. None of the alternative rows "
              "inherits the current frame's clearance or simulated walking evidence.",
              "- Shortlist walking joints by loaded torque **at the required speed and duty**, thermal rise in "
              "the intended mounting, mass including transmission/support, and full-body supply load.", ""]
    (ROOT / "docs/research/ape-actuator-landscape.md").write_text("\n".join(lines))
    print(f"{len(ROWS)} leads in {len(counts)} groups")


if __name__ == "__main__":
    write()
