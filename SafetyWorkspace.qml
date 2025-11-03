/*
 ═ ═*═════════════════════════════════════════════════════════════
 FILE: qml/components/SafetyWorkspace.qml
 ABHAYA-AI: PROFESSIONAL AUTOMOTIVE SAFETY INTERFACE
 ✅ SYSTEMATIC UI DESIGN SYSTEM

 DESIGN SYSTEM COLOR ASSIGNMENT:
 ───────────────────────────────────────────────────────────────
 TYPOGRAPHY:
 - Primary Titles     → Bright Cyan (#00FFFF)
 - Secondary Titles   → Electric Blue (#0080FF)
 - Body Text          → Light Gray (#CCCCCC)
 - Subtle Text        → Medium Gray (#888888)
 - Hint Text          → Dark Gray (#555555)

 BACKGROUNDS:
 - Cards              → Dark Blue (#161B24)
 - Panels             → Darker (#0D1117)
 - Sections           → Deep Teal tint

 BORDERS & DIVIDERS:
 - Primary Borders    → Bright Cyan (#00FFFF)
 - Card Borders       → Electric Blue (#0080FF)
 - Dividers           → Deep Teal (#003D4D)

 INTERACTIVE ELEMENTS:
 - Primary Buttons    → Mint Green (#00FFA3)
 - Secondary Actions  → Electric Blue (#0080FF)
 - Hover States       → Electric Blue @ 30%
 - Active States      → Mint Green

 DATA VISUALIZATION:
 - Positive/Good      → Mint Green (#00FFA3)
 - Neutral/Info       → Bright Cyan (#00FFFF)
 - Warning            → Electric Blue (#0080FF)
 - Critical/Alert     → Red (#FF4466)
 ═══════════════════════════════════════════════════════════════
 */

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    width: 1024
    height: 600
    color: "#0A0E14"

    signal closeClicked()

    // ═══════════════════════════════════════════════════════════
    // DESIGN SYSTEM - COLOR CONSTANTS
    // ═══════════════════════════════════════════════════════════

    // Primary Palette
    readonly property color colorBrandPrimary: "#00FFFF"      // Bright Cyan
    readonly property color colorBrandSecondary: "#0080FF"    // Electric Blue
    readonly property color colorPositive: "#00FFA3"          // Mint Green
    readonly property color colorDepth: "#003D4D"             // Deep Teal
    readonly property color colorCritical: "#FF4466"          // Red

    // Typography Colors
    readonly property color colorTitlePrimary: "#00FFFF"      // Bright Cyan - Main titles
    readonly property color colorTitleSecondary: "#0080FF"    // Electric Blue - Subtitles
    readonly property color colorBodyText: "#CCCCCC"          // Light Gray - Main content
    readonly property color colorSubtleText: "#888888"        // Medium Gray - Labels
    readonly property color colorHintText: "#555555"          // Dark Gray - Placeholders

    // Background Colors
    readonly property color colorCardBg: "#161B24"            // Dark Blue - Cards
    readonly property color colorPanelBg: "#0D1117"           // Darker - Panels
    readonly property color colorSectionBg: "#1A1F2E"         // Section backgrounds

    // Border Colors
    readonly property color colorBorderPrimary: "#00FFFF"     // Bright Cyan - Main borders
    readonly property color colorBorderSecondary: "#0080FF"   // Electric Blue - Card borders
    readonly property color colorDivider: "#003D4D"           // Deep Teal - Dividers

    // ═══════════════════════════════════════════════════════════
    // DATA PROPERTIES
    // ═══════════════════════════════════════════════════════════
    property int wellnessScore: 87
    property int fatigueLevel: 28
    property int hrvScore: 82
    property int stressLevel: 22
    property int alertnessScore: 91
    property int heartRate: 68
    property int spo2: 98
    property int respirationRate: 16
    property int cabinCO2: 720
    property real temperature: 21.8
    property int humidity: 42
    property int airQuality: 94
    property string tripTime: "1:47"
    property string lastBreak: "28min"

    property bool settingsOpen: false

    // System status
    property bool steeringActive: true
    property bool voiceActive: true
    property bool hrvActive: true
    property bool microSleepActive: true
    property bool aiCompanionActive: true

    // Intervention data
    property int currentInterventionLevel: 0
    property string lastIntervention: "None"
    property int interventionsToday: 3
    property string nextBreakSuggestion: "15 min"

    // Mock updates
    Timer {
        interval: 4000
        running: true
        repeat: true
        onTriggered: {
            wellnessScore = Math.min(100, Math.max(50, wellnessScore + Math.floor(Math.random() * 10 - 5)))
            fatigueLevel = Math.min(100, Math.max(0, fatigueLevel + Math.floor(Math.random() * 12 - 6)))
            hrvScore = Math.min(100, Math.max(60, hrvScore + Math.floor(Math.random() * 8 - 4)))
            stressLevel = Math.min(80, Math.max(10, stressLevel + Math.floor(Math.random() * 10 - 5)))
            alertnessScore = Math.min(100, Math.max(60, alertnessScore + Math.floor(Math.random() * 8 - 4)))
            heartRate = Math.min(85, Math.max(60, heartRate + Math.floor(Math.random() * 4 - 2)))
            spo2 = Math.min(100, Math.max(94, spo2 + Math.floor(Math.random() * 2 - 1)))
            respirationRate = Math.min(20, Math.max(12, respirationRate + Math.floor(Math.random() * 2 - 1)))
            cabinCO2 = Math.min(1100, Math.max(500, cabinCO2 + Math.floor(Math.random() * 80 - 40)))
            temperature = Math.min(26, Math.max(19, temperature + (Math.random() * 0.8 - 0.4)))
            humidity = Math.min(65, Math.max(35, humidity + Math.floor(Math.random() * 6 - 3)))
            airQuality = Math.min(100, Math.max(70, airQuality + Math.floor(Math.random() * 8 - 4)))
        }
    }

    // ═══════════════════════════════════════════════════════════
    // BACKGROUND
    // ═══════════════════════════════════════════════════════════
    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            orientation: Gradient.Vertical
            GradientStop { position: 0.0; color: "#0F1419" }
            GradientStop { position: 1.0; color: "#0A0E14" }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MAIN DASHBOARD
    // ═══════════════════════════════════════════════════════════
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 8

        // ═══════════════════════════════════════════════════════
        // HEADER
        // ═══════════════════════════════════════════════════════
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 50
            radius: 10
            color: root.colorCardBg + "E0"  // Card background
            border.color: root.colorBorderPrimary + "40"  // Primary border (subtle)
            border.width: 1

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 15
                anchors.rightMargin: 15
                spacing: 12

                // Logo + Title
                RowLayout {
                    spacing: 8

                    Rectangle {
                        width: 36
                        height: 36
                        radius: 18
                        color: root.colorBrandPrimary + "20"  // Brand primary glow
                        border.color: root.colorBrandPrimary  // Brand primary border
                        border.width: 2

                        Text {
                            anchors.centerIn: parent
                            text: "🛡️"
                            font.pixelSize: 20
                        }
                    }

                    ColumnLayout {
                        spacing: -2

                        Text {
                            text: "ABHAYA-AI"
                            font.pixelSize: 14
                            font.weight: Font.Bold
                            font.letterSpacing: 1.2
                            color: root.colorTitlePrimary  // Primary title color
                        }

                        Text {
                            text: "Driver Wellness System"
                            font.pixelSize: 8
                            color: root.colorSubtleText  // Subtle text
                        }
                    }
                }

                Item { Layout.fillWidth: true }

                // System Status Dots
                RowLayout {
                    spacing: 6

                    StatusDot {
                        active: root.steeringActive
                        tooltip: "Steering"
                        onClicked: root.steeringActive = !root.steeringActive
                    }
                    StatusDot {
                        active: root.voiceActive
                        tooltip: "Voice"
                        onClicked: root.voiceActive = !root.voiceActive
                    }
                    StatusDot {
                        active: root.hrvActive
                        tooltip: "HRV"
                        onClicked: root.hrvActive = !root.hrvActive
                    }
                    StatusDot {
                        active: root.microSleepActive
                        tooltip: "Sleep"
                        onClicked: root.microSleepActive = !root.microSleepActive
                    }
                    StatusDot {
                        active: root.aiCompanionActive
                        tooltip: "AI"
                        onClicked: root.aiCompanionActive = !root.aiCompanionActive
                    }
                }

                Rectangle {
                    width: 1
                    height: 30
                    color: root.colorDivider + "60"  // Divider color
                }

                // Settings Button
                Rectangle {
                    width: 36
                    height: 36
                    radius: 18
                    color: settingsBtn.containsMouse ? root.colorBrandSecondary + "30" : "transparent"  // Hover: secondary color
                    border.color: root.colorBorderPrimary  // Primary border
                    border.width: 2

                    Behavior on color { ColorAnimation { duration: 200 } }

                    Text {
                        anchors.centerIn: parent
                        text: root.settingsOpen ? "✕" : "⚙"
                        font.pixelSize: 18
                        color: root.colorTitlePrimary  // Primary title color
                        rotation: root.settingsOpen ? 90 : 0
                        Behavior on rotation {
                            NumberAnimation { duration: 300; easing.type: Easing.OutCubic }
                        }
                    }

                    MouseArea {
                        id: settingsBtn
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: {
                            console.log("⚙️ Settings toggled:", !root.settingsOpen)
                            root.settingsOpen = !root.settingsOpen
                        }
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════
        // MAIN CONTENT GRID
        // ═══════════════════════════════════════════════════════
        GridLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            columns: 3
            rows: 2
            rowSpacing: 8
            columnSpacing: 8

            // FATIGUE ANALYSIS (with Wellness Arc)
            CompactCard {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.rowSpan: 2
                title: "FATIGUE ANALYSIS"
                icon: "📊"
                accentColor: root.colorBorderSecondary  // Secondary border

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 8

                    // Wellness Progress Arc
                    Item {
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        CircularGauge {
                            anchors.centerIn: parent
                            width: Math.min(parent.width, parent.height) * 0.85
                            height: width
                            value: root.wellnessScore
                            maxValue: 100
                            // Data viz: Good=Positive, Medium=Info, Bad=Critical
                            gaugeColor: root.wellnessScore > 75 ? root.colorPositive :
                            (root.wellnessScore > 50 ? root.colorBrandPrimary : root.colorCritical)
                            backgroundColor: root.colorDepth  // Depth color for background
                            lineWidth: 10
                        }

                        ColumnLayout {
                            anchors.centerIn: parent
                            spacing: 2

                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: root.wellnessScore + "%"
                                font.pixelSize: 38
                                font.weight: Font.Bold
                                // Data viz color
                                color: root.wellnessScore > 75 ? root.colorPositive :
                                (root.wellnessScore > 50 ? root.colorBrandPrimary : root.colorCritical)
                            }

                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: "WELLNESS"
                                font.pixelSize: 12
                                font.weight: Font.Bold
                                font.letterSpacing: 1.0
                                color: root.colorTitlePrimary  // Primary title
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        MetricBar {
                            Layout.fillWidth: true
                            label: "Alertness"
                            value: root.alertnessScore
                            barColor: root.colorPositive  // Positive = good
                        }

                        MetricBar {
                            Layout.fillWidth: true
                            label: "Stress"
                            value: root.stressLevel
                            barColor: root.stressLevel > 50 ? root.colorCritical : root.colorPositive  // Critical if high
                        }

                        MetricBar {
                            Layout.fillWidth: true
                            label: "HRV Score"
                            value: root.hrvScore
                            barColor: root.colorPositive  // Positive = good
                        }

                        MetricBar {
                            Layout.fillWidth: true
                            label: "Blood O₂"
                            value: root.spo2
                            barColor: root.colorPositive  // Positive = good
                        }

                        MetricBar {
                            Layout.fillWidth: true
                            label: "Respiration"
                            value: root.respirationRate
                            barColor: root.colorBrandPrimary  // Info color
                        }
                    }
                }
            }

            // BIOMETRICS
            CompactCard {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "BIOMETRICS"
                icon: "💓"
                accentColor: root.colorBorderSecondary  // Secondary border

                GridLayout {
                    anchors.fill: parent
                    columns: 2
                    rowSpacing: 12
                    columnSpacing: 12

                    DataPoint {
                        Layout.fillWidth: true
                        label: "Heart Rate"
                        value: root.heartRate
                        unit: "BPM"
                        valueColor: root.colorBrandSecondary  // Secondary color
                    }

                    DataPoint {
                        Layout.fillWidth: true
                        label: "HRV Score"
                        value: root.hrvScore
                        unit: "/100"
                        valueColor: root.colorPositive  // Positive = good
                    }

                    DataPoint {
                        Layout.fillWidth: true
                        label: "Stress"
                        value: root.stressLevel
                        unit: "%"
                        valueColor: root.stressLevel > 50 ? root.colorCritical : root.colorPositive
                    }

                    DataPoint {
                        Layout.fillWidth: true
                        label: "Alertness"
                        value: root.alertnessScore
                        unit: "/100"
                        valueColor: root.colorPositive  // Positive = good
                    }
                }
            }

            // CABIN ENVIRONMENT
            CompactCard {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "CABIN ENVIRONMENT"
                icon: "🌡️"
                accentColor: root.colorBorderSecondary  // Secondary border

                GridLayout {
                    anchors.fill: parent
                    columns: 2
                    rowSpacing: 12
                    columnSpacing: 12

                    DataPoint {
                        Layout.fillWidth: true
                        label: "CO₂"
                        value: root.cabinCO2
                        unit: "ppm"
                        valueColor: root.cabinCO2 > 1000 ? root.colorCritical : root.colorBrandPrimary
                        alert: root.cabinCO2 > 1000
                    }

                    DataPoint {
                        Layout.fillWidth: true
                        label: "Temp"
                        value: root.temperature.toFixed(1)
                        unit: "°C"
                        valueColor: root.colorBrandPrimary  // Info color
                    }

                    DataPoint {
                        Layout.fillWidth: true
                        label: "Humidity"
                        value: root.humidity
                        unit: "%"
                        valueColor: root.colorBrandPrimary  // Info color
                    }

                    DataPoint {
                        Layout.fillWidth: true
                        label: "Air Quality"
                        value: root.airQuality
                        unit: "/100"
                        valueColor: root.colorPositive  // Positive = good
                    }
                }
            }

            // TRIP DATA
            CompactCard {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "TRIP DATA"
                icon: "🚗"
                accentColor: root.colorBorderSecondary  // Secondary border

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2

                            Text {
                                text: "Duration"
                                font.pixelSize: 10
                                color: root.colorSubtleText  // Subtle text for labels
                            }

                            Text {
                                text: root.tripTime
                                font.pixelSize: 26
                                font.weight: Font.Bold
                                color: root.colorPositive  // Positive color
                            }
                        }

                        Rectangle {
                            width: 1
                            height: 40
                            color: root.colorDivider + "60"  // Divider
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2

                            Text {
                                text: "Last Break"
                                font.pixelSize: 10
                                color: root.colorSubtleText  // Subtle text for labels
                            }

                            Text {
                                text: root.lastBreak
                                font.pixelSize: 26
                                font.weight: Font.Bold
                                color: root.colorBrandPrimary  // Info color
                            }
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 1
                        color: root.colorDivider + "60"  // Divider
                    }

                    ActionBtn {
                        Layout.fillWidth: true
                        label: "Find Rest Stop"
                        icon: "📍"
                        btnColor: root.colorPositive  // Primary action = positive
                    }
                }
            }

            // ACTIVE INTERVENTIONS
            CompactCard {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "ACTIVE INTERVENTIONS"
                icon: "⚡"
                accentColor: root.colorBorderSecondary  // Secondary border

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 55
                        radius: 10
                        // Data viz colors based on level
                        color: root.currentInterventionLevel === 0 ? root.colorPositive + "15" :
                        (root.currentInterventionLevel < 3 ? root.colorBrandPrimary + "20" : root.colorCritical + "15")
                        border.color: root.currentInterventionLevel === 0 ? root.colorPositive :
                        (root.currentInterventionLevel < 3 ? root.colorBrandPrimary : root.colorCritical)
                        border.width: 2

                        ColumnLayout {
                            anchors.centerIn: parent
                            spacing: 2

                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: "Level " + root.currentInterventionLevel
                                font.pixelSize: 20
                                font.weight: Font.Bold
                                color: root.currentInterventionLevel === 0 ? root.colorPositive :
                                (root.currentInterventionLevel < 3 ? root.colorBrandPrimary : root.colorCritical)
                            }

                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: root.currentInterventionLevel === 0 ? "All Systems Normal" : "Active Warning"
                                font.pixelSize: 8
                                color: root.colorSubtleText  // Subtle text
                            }
                        }
                    }

                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        rowSpacing: 8
                        columnSpacing: 8

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2

                            Text {
                                text: "Today"
                                font.pixelSize: 9
                                color: root.colorSubtleText  // Subtle text
                            }

                            Text {
                                text: root.interventionsToday.toString()
                                font.pixelSize: 22
                                font.weight: Font.Bold
                                color: root.colorBrandSecondary  // Secondary color
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2

                            Text {
                                text: "Next Break"
                                font.pixelSize: 9
                                color: root.colorSubtleText  // Subtle text
                            }

                            Text {
                                text: root.nextBreakSuggestion
                                font.pixelSize: 22
                                font.weight: Font.Bold
                                color: root.colorPositive  // Positive color
                            }
                        }
                    }

                    ActionBtn {
                        Layout.fillWidth: true
                        label: "View History"
                        icon: "📋"
                        btnColor: root.colorBrandSecondary  // Secondary action
                    }
                }
            }
        }

        // SWIPE TO OPEN
        MouseArea {
            id: mainSwipeArea
            Layout.fillWidth: true
            Layout.fillHeight: true
            enabled: !root.settingsOpen
            z: -1

            property real startX: 0

            onPressed: (mouse) => {
                startX = mouse.x
            }

            onReleased: (mouse) => {
                var deltaX = mouse.x - startX
                if (deltaX < -150) {
                    console.log("➡️ Opening settings")
                    root.settingsOpen = true
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // SETTINGS OVERLAY
    // ═══════════════════════════════════════════════════════════
    MouseArea {
        id: settingsOverlay
        anchors.fill: parent
        visible: root.settingsOpen
        z: 98

        Rectangle {
            anchors.fill: parent
            color: "#000000"
            opacity: 0.7
        }

        onClicked: {
            console.log("🖱️ Overlay clicked - closing")
            root.settingsOpen = false
        }
    }

    // ═══════════════════════════════════════════════════════════
    // SETTINGS PANEL
    // ═══════════════════════════════════════════════════════════
    Item {
        id: settingsPanelContainer
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        width: 420
        visible: root.settingsOpen
        z: 99

        transform: Translate {
            id: slideTransform
            x: root.settingsOpen ? 0 : 420

            Behavior on x {
                NumberAnimation {
                    duration: 300
                    easing.type: Easing.OutCubic
                }
            }
        }

        Rectangle {
            anchors.fill: parent
            color: root.colorPanelBg  // Panel background
            border.color: root.colorBorderPrimary  // Primary border
            border.width: 2

            ColumnLayout {
                anchors.fill: parent
                spacing: 0

                // Header
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 60
                    color: root.colorCardBg  // Card background
                    border.color: root.colorBorderPrimary  // Primary border
                    border.width: 1

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 20
                        anchors.rightMargin: 20

                        Text {
                            text: "⚙ AI CONFIGURATION"
                            font.pixelSize: 14
                            font.weight: Font.Bold
                            font.letterSpacing: 1.2
                            color: root.colorTitlePrimary  // Primary title
                            Layout.fillWidth: true
                        }

                        Rectangle {
                            width: 40
                            height: 40
                            radius: 20
                            color: closeBtn.pressed ? root.colorCritical + "60" : (closeBtn.containsMouse ? root.colorCritical + "30" : "transparent")
                            border.color: root.colorCritical  // Critical color for close
                            border.width: 2

                            Behavior on color { ColorAnimation { duration: 150 } }

                            Text {
                                anchors.centerIn: parent
                                text: "✕"
                                font.pixelSize: 24
                                font.weight: Font.Bold
                                color: root.colorCritical  // Critical color
                            }

                            MouseArea {
                                id: closeBtn
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor

                                onClicked: {
                                    console.log("❌ X clicked - closing")
                                    root.settingsOpen = false
                                }
                            }
                        }
                    }
                }

                // Content
                Flickable {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    contentHeight: settingsContent.height
                    clip: true
                    boundsBehavior: Flickable.StopAtBounds

                    ScrollBar.vertical: ScrollBar {
                        width: 6
                        policy: ScrollBar.AsNeeded
                        contentItem: Rectangle {
                            radius: 3
                            color: root.colorBrandPrimary  // Primary color
                            opacity: 0.6
                        }
                    }

                    ColumnLayout {
                        id: settingsContent
                        width: parent.width - 30
                        anchors.horizontalCenter: parent.horizontalCenter
                        spacing: 15

                        Item { Layout.preferredHeight: 15 }

                        SettingsBlock {
                            Layout.fillWidth: true
                            title: "Detection Systems"

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Steering Analysis (IMU)"
                                    checked: root.steeringActive
                                    onToggled: (state) => { root.steeringActive = state }
                                }

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Voice Biomarker Detection"
                                    checked: root.voiceActive
                                    onToggled: (state) => { root.voiceActive = state }
                                }

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "HRV Monitor (Earlobe)"
                                    checked: root.hrvActive
                                    onToggled: (state) => { root.hrvActive = state }
                                }

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Micro-Sleep Detector"
                                    checked: root.microSleepActive
                                    onToggled: (state) => { root.microSleepActive = state }
                                }

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "AI Companion System"
                                    checked: root.aiCompanionActive
                                    onToggled: (state) => { root.aiCompanionActive = state }
                                }
                            }
                        }

                        SettingsBlock {
                            Layout.fillWidth: true
                            title: "Detection Sensitivity"

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 10

                                Text {
                                    text: "Fatigue Detection Threshold"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText  // Subtle text
                                }

                                CompactSlider {
                                    Layout.fillWidth: true
                                    value: 60
                                    from: 30
                                    to: 90
                                    unit: "%"
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Micro-Sleep Sensitivity"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                TriSwitch {
                                    Layout.fillWidth: true
                                    options: ["Low", "Medium", "High"]
                                    selected: 1
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Voice Stress Threshold"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                CompactSlider {
                                    Layout.fillWidth: true
                                    value: 70
                                    from: 40
                                    to: 90
                                    unit: "%"
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Steering Pattern Analysis"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                TriSwitch {
                                    Layout.fillWidth: true
                                    options: ["Lenient", "Normal", "Strict"]
                                    selected: 1
                                }
                            }
                        }

                        SettingsBlock {
                            Layout.fillWidth: true
                            title: "Alert Thresholds"

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 10

                                Text {
                                    text: "Heart Rate Alert (High)"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                CompactSlider {
                                    Layout.fillWidth: true
                                    value: 100
                                    from: 85
                                    to: 120
                                    unit: "BPM"
                                }

                                SettingsDivider {}

                                Text {
                                    text: "CO₂ Warning Level"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                CompactSlider {
                                    Layout.fillWidth: true
                                    value: 1000
                                    from: 600
                                    to: 1500
                                    unit: "ppm"
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Blood O₂ Low Threshold"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                CompactSlider {
                                    Layout.fillWidth: true
                                    value: 94
                                    from: 88
                                    to: 96
                                    unit: "%"
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Stress Level Alert"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                CompactSlider {
                                    Layout.fillWidth: true
                                    value: 75
                                    from: 50
                                    to: 90
                                    unit: "%"
                                }
                            }
                        }

                        SettingsBlock {
                            Layout.fillWidth: true
                            title: "Intervention Preferences"

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                Text {
                                    text: "Enable Intervention Levels"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                InterventionLevel { level: 1; name: "Ambient (Light/Temp)"; checked: true }
                                InterventionLevel { level: 2; name: "Audio Alerts"; checked: true }
                                InterventionLevel { level: 3; name: "Haptic Vibration"; checked: true }
                                InterventionLevel { level: 4; name: "Visual Warnings"; checked: true }
                                InterventionLevel { level: 5; name: "Emergency Protocol"; checked: true }

                                SettingsDivider {}

                                Text {
                                    text: "Intervention Escalation Time"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                CompactSlider {
                                    Layout.fillWidth: true
                                    value: 30
                                    from: 10
                                    to: 60
                                    unit: "sec"
                                }
                            }
                        }

                        SettingsBlock {
                            Layout.fillWidth: true
                            title: "AI Companion Behavior"

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Enable Ghost Passenger AI"
                                    checked: true
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Conversation Frequency"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                TriSwitch {
                                    Layout.fillWidth: true
                                    options: ["Rare", "Normal", "Frequent"]
                                    selected: 1
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Engagement Style"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                TriSwitch {
                                    Layout.fillWidth: true
                                    options: ["Casual", "Balanced", "Professional"]
                                    selected: 1
                                }

                                SettingsDivider {}

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Auto-Start on Fatigue Detection"
                                    checked: true
                                }

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Enable Alertness Games"
                                    checked: true
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Game Difficulty"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                TriSwitch {
                                    Layout.fillWidth: true
                                    options: ["Easy", "Medium", "Hard"]
                                    selected: 1
                                }
                            }
                        }

                        SettingsBlock {
                            Layout.fillWidth: true
                            title: "Break Management"

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                Text {
                                    text: "Auto-Suggest Break Every"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                CompactSlider {
                                    Layout.fillWidth: true
                                    value: 90
                                    from: 45
                                    to: 180
                                    unit: "min"
                                }

                                SettingsDivider {}

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Predictive Break Recommendations"
                                    checked: true
                                }

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Auto Find Rest Stops"
                                    checked: true
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Route Replanning Threshold"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                TriSwitch {
                                    Layout.fillWidth: true
                                    options: ["Moderate", "High", "Critical"]
                                    selected: 1
                                }
                            }
                        }

                        SettingsBlock {
                            Layout.fillWidth: true
                            title: "Ambient Intelligence"

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Auto HVAC Adjustment"
                                    checked: true
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Target Temp on Drowsiness"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                CompactSlider {
                                    Layout.fillWidth: true
                                    value: 19
                                    from: 16
                                    to: 22
                                    unit: "°C"
                                }

                                SettingsDivider {}

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Adaptive Cabin Lighting"
                                    checked: true
                                }

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Air Quality Auto-Optimization"
                                    checked: true
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Fresh Air Injection Trigger"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                CompactSlider {
                                    Layout.fillWidth: true
                                    value: 1000
                                    from: 700
                                    to: 1300
                                    unit: "ppm"
                                }
                            }
                        }

                        SettingsBlock {
                            Layout.fillWidth: true
                            title: "Learning & Personalization"

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Drowsiness Pattern Learning"
                                    checked: true
                                }

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Adaptive Threshold Adjustment"
                                    checked: true
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Learning Aggressiveness"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                TriSwitch {
                                    Layout.fillWidth: true
                                    options: ["Conservative", "Normal", "Aggressive"]
                                    selected: 1
                                }

                                SettingsDivider {}

                                SmallButton {
                                    Layout.fillWidth: true
                                    label: "Calibrate Baseline (5 min)"
                                    icon: "🎯"
                                }

                                SmallButton {
                                    Layout.fillWidth: true
                                    label: "Reset Learning Data"
                                    icon: "🔄"
                                }
                            }
                        }

                        SettingsBlock {
                            Layout.fillWidth: true
                            title: "Emergency Settings"

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Emergency Alert System"
                                    checked: true
                                }

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Passenger Safety Co-Pilot"
                                    checked: true
                                }

                                SettingsDivider {}

                                SmallButton {
                                    Layout.fillWidth: true
                                    label: "Configure Emergency Contacts"
                                    icon: "📱"
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Critical Alert Trigger Level"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                TriSwitch {
                                    Layout.fillWidth: true
                                    options: ["Level 3", "Level 4", "Level 5"]
                                    selected: 2
                                }
                            }
                        }

                        SettingsBlock {
                            Layout.fillWidth: true
                            title: "Data & Privacy"

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                Rectangle {
                                    Layout.fillWidth: true
                                    height: 32
                                    radius: 6
                                    color: root.colorBrandPrimary + "10"  // Primary glow
                                    border.color: root.colorBrandPrimary  // Primary border
                                    border.width: 1

                                    Text {
                                        anchors.centerIn: parent
                                        text: "🔒 All processing done locally"
                                        font.pixelSize: 9
                                        color: root.colorTitlePrimary  // Primary title
                                    }
                                }

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Enable Trip History Logging"
                                    checked: true
                                }

                                SettingSwitch {
                                    Layout.fillWidth: true
                                    label: "Share Data with Fleet Dashboard"
                                    checked: false
                                }

                                SettingsDivider {}

                                Text {
                                    text: "Data Retention Period"
                                    font.pixelSize: 9
                                    color: root.colorSubtleText
                                }

                                TriSwitch {
                                    Layout.fillWidth: true
                                    options: ["7 days", "30 days", "90 days"]
                                    selected: 1
                                }

                                SettingsDivider {}

                                SmallButton {
                                    Layout.fillWidth: true
                                    label: "Export Trip Data"
                                    icon: "📤"
                                }

                                SmallButton {
                                    Layout.fillWidth: true
                                    label: "Clear All Data"
                                    icon: "🗑️"
                                }
                            }
                        }

                        Item { Layout.preferredHeight: 15 }
                    }

                    MouseArea {
                        anchors.fill: parent
                        z: -1

                        property real startX: 0
                        property real startY: 0

                        onPressed: (mouse) => {
                            startX = mouse.x
                            startY = mouse.y
                            mouse.accepted = false
                        }

                        onReleased: (mouse) => {
                            var deltaX = mouse.x - startX
                            var deltaY = mouse.y - startY

                            if (Math.abs(deltaX) > Math.abs(deltaY) && deltaX > 150) {
                                console.log("⬅️ Closing settings")
                                root.settingsOpen = false
                                mouse.accepted = true
                            } else {
                                mouse.accepted = false
                            }
                        }
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // COMPONENTS (ALL USE DESIGN SYSTEM COLORS)
    // ═══════════════════════════════════════════════════════════

    component StatusDot: Rectangle {
        property bool active: true
        property string tooltip: ""
        signal clicked()

        width: 8
        height: 8
        radius: 4
        color: active ? root.colorPositive : root.colorCritical  // Positive/Critical

        SequentialAnimation on opacity {
            running: active
            loops: Animation.Infinite
            NumberAnimation { to: 0.4; duration: 800 }
            NumberAnimation { to: 1.0; duration: 800 }
        }

        MouseArea {
            anchors.fill: parent
            anchors.margins: -4
            cursorShape: Qt.PointingHandCursor
            onClicked: parent.clicked()
        }
    }

    component CircularGauge: Canvas {
        property int value: 0
        property int maxValue: 100
        property color gaugeColor: root.colorBrandPrimary
        property color backgroundColor: root.colorDepth
        property int lineWidth: 10

        antialiasing: true
        renderStrategy: Canvas.Threaded
        renderTarget: Canvas.FramebufferObject

        onValueChanged: requestPaint()
        onLineWidthChanged: requestPaint()

        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()

            var centerX = width / 2
            var centerY = height / 2
            var radius = Math.min(width, height) / 2 - lineWidth

            ctx.beginPath()
            ctx.arc(centerX, centerY, radius, 0.75 * Math.PI, 2.25 * Math.PI, false)
            ctx.lineWidth = lineWidth
            ctx.strokeStyle = backgroundColor
            ctx.stroke()

            var endAngle = 0.75 * Math.PI + (value / maxValue) * 1.5 * Math.PI

            ctx.beginPath()
            ctx.arc(centerX, centerY, radius, 0.75 * Math.PI, endAngle, false)
            ctx.lineWidth = lineWidth
            ctx.strokeStyle = gaugeColor
            ctx.lineCap = "round"
            ctx.stroke()
        }
    }

    component CompactCard: Rectangle {
        property string title: ""
        property string icon: ""
        property color accentColor: root.colorBorderSecondary
        default property alias content: contentArea.data

            radius: 10
            color: root.colorCardBg + "E0"  // Card background
            border.color: accentColor
            border.width: 2

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Text {
                        text: icon
                        font.pixelSize: 14
                    }

                    Text {
                        text: title
                        font.pixelSize: 10
                        font.weight: Font.Bold
                        font.letterSpacing: 0.8
                        color: root.colorTitlePrimary  // Primary title
                        Layout.fillWidth: true
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    height: 2
                    color: accentColor + "60"  // Accent underline
                }

                Item {
                    id: contentArea
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
            }
    }

    component DataPoint: ColumnLayout {
        property string label: ""
        property var value: 0
        property string unit: ""
        property color valueColor: root.colorBrandPrimary
        property bool alert: false

        spacing: 2

        Text {
            text: label
            font.pixelSize: 10
            color: root.colorSubtleText  // Subtle text for labels
        }

        RowLayout {
            spacing: 4

            Text {
                text: value.toString()
                font.pixelSize: 26
                font.weight: Font.Bold
                color: parent.parent.valueColor
            }

            Text {
                text: unit
                font.pixelSize: 10
                color: root.colorHintText  // Hint text for units
                Layout.alignment: Qt.AlignBottom
                Layout.bottomMargin: 3
            }

            Rectangle {
                width: 6
                height: 6
                radius: 3
                color: root.colorCritical  // Critical for alerts
                visible: parent.parent.alert

                SequentialAnimation on opacity {
                    running: parent.parent.alert
                    loops: Animation.Infinite
                    NumberAnimation { to: 0.3; duration: 500 }
                    NumberAnimation { to: 1.0; duration: 500 }
                }
            }
        }
    }

    component MetricBar: RowLayout {
        property string label: ""
        property int value: 0
        property color barColor: root.colorBrandPrimary

        spacing: 8

        Text {
            text: label
            font.pixelSize: 11
            font.weight: Font.Medium
            color: root.colorSubtleText  // Subtle text for labels
            Layout.preferredWidth: 70
        }

        Rectangle {
            Layout.fillWidth: true
            height: 10
            radius: 5
            color: root.colorSectionBg  // Section background
            border.color: parent.barColor + "40"
            border.width: 1

            Rectangle {
                width: (parent.width * parent.parent.value) / 100
                height: parent.height
                radius: 5
                color: parent.parent.barColor

                Behavior on width {
                    NumberAnimation { duration: 400; easing.type: Easing.OutCubic }
                }
            }
        }

        Text {
            text: value + "%"
            font.pixelSize: 11
            font.weight: Font.Bold
            color: parent.barColor
            Layout.preferredWidth: 40
        }
    }

    component ActionBtn: Rectangle {
        property string label: ""
        property string icon: ""
        property color btnColor: root.colorPositive

        height: 32
        radius: 8
        color: mouse.containsMouse ? btnColor + "30" : btnColor + "10"  // Hover state
        border.color: btnColor
        border.width: 1

        Behavior on color { ColorAnimation { duration: 200 } }

        RowLayout {
            anchors.centerIn: parent
            spacing: 6

            Text {
                text: icon
                font.pixelSize: 14
            }

            Text {
                text: label
                font.pixelSize: 10
                font.weight: Font.Medium
                color: parent.parent.btnColor
            }
        }

        MouseArea {
            id: mouse
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: Qt.PointingHandCursor
            onClicked: console.log(label)
        }
    }

    component SettingsBlock: ColumnLayout {
        property string title: ""
        default property alias content: contentArea.data

            spacing: 8

            Text {
                text: title
                font.pixelSize: 11
                font.weight: Font.Bold
                color: root.colorTitleSecondary  // Secondary title
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: root.colorDivider + "60"  // Divider
            }

            ColumnLayout {
                id: contentArea
                Layout.fillWidth: true
                spacing: 6
            }
    }

    component SettingSwitch: RowLayout {
        property string label: ""
        property bool checked: false
        signal toggled(bool state)

        spacing: 8

        Text {
            Layout.fillWidth: true
            text: label
            font.pixelSize: 9
            color: root.colorBodyText  // Body text
            wrapMode: Text.WordWrap
        }

        Rectangle {
            width: 38
            height: 20
            radius: 10
            color: parent.checked ? root.colorBrandSecondary : "#2A2A2A"  // Secondary when active

            Behavior on color { ColorAnimation { duration: 200 } }

            Rectangle {
                x: parent.parent.checked ? parent.width - width - 2 : 2
                anchors.verticalCenter: parent.verticalCenter
                width: 16
                height: 16
                radius: 8
                color: "#FFFFFF"

                Behavior on x {
                    NumberAnimation { duration: 200; easing.type: Easing.OutCubic }
                }
            }

            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: {
                    parent.parent.checked = !parent.parent.checked
                    parent.parent.toggled(parent.parent.checked)
                }
            }
        }
    }

    component TriSwitch: Item {
        property var options: []
        property int selected: 0

        implicitHeight: 40

        Rectangle {
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: parent.top
            anchors.topMargin: 8
            width: parent.width - 20
            height: 2
            radius: 1
            color: "#2A2A2A"

            Rectangle {
                width: (parent.width / 2) * (parent.parent.selected / 1.0)
                height: parent.height
                color: root.colorBrandSecondary  // Secondary for progress
                radius: 1

                Behavior on width {
                    NumberAnimation { duration: 250; easing.type: Easing.OutCubic }
                }
            }
        }

        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: parent.top
            anchors.topMargin: 4
            spacing: (parent.width - 70) / 2

            Repeater {
                model: 3
                Rectangle {
                    width: 12
                    height: 12
                    radius: 6
                    color: index === parent.parent.parent.selected ? root.colorBrandSecondary : "#2A2A2A"
                    border.color: root.colorBrandSecondary
                    border.width: 1

                    Behavior on color { ColorAnimation { duration: 200 } }

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: parent.parent.parent.parent.selected = index
                    }
                }
            }
        }

        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 2
            spacing: 10

            Repeater {
                model: parent.parent.options
                Text {
                    text: modelData
                    font.pixelSize: 7
                    font.weight: index === parent.parent.parent.selected ? Font.Bold : Font.Normal
                    color: index === parent.parent.parent.selected ? root.colorBrandSecondary : root.colorHintText

                    Behavior on color { ColorAnimation { duration: 200 } }
                }
            }
        }
    }

    component InterventionLevel: RowLayout {
        property int level: 1
        property string name: ""
        property bool checked: false

        spacing: 8

        Rectangle {
            width: 20
            height: 20
            radius: 4
            color: parent.checked ? root.colorBrandSecondary + "20" : "transparent"
            border.color: root.colorBrandSecondary
            border.width: 1

            Text {
                anchors.centerIn: parent
                text: level
                font.pixelSize: 10
                font.weight: Font.Bold
                color: parent.parent.checked ? root.colorBrandSecondary : root.colorHintText
            }
        }

        Text {
            Layout.fillWidth: true
            text: name
            font.pixelSize: 9
            color: root.colorBodyText  // Body text
        }

        Rectangle {
            width: 16
            height: 16
            radius: 8
            color: parent.checked ? root.colorBrandSecondary : "transparent"
            border.color: root.colorBrandSecondary
            border.width: 2

            Rectangle {
                anchors.centerIn: parent
                width: 8
                height: 8
                radius: 4
                color: root.colorBrandSecondary
                visible: parent.parent.checked
            }

            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: parent.parent.checked = !parent.parent.checked
            }
        }
    }

    component CompactSlider: RowLayout {
        property real value: 50
        property real from: 0
        property real to: 100
        property string unit: ""

        spacing: 8

        Slider {
            id: slider
            Layout.fillWidth: true
            from: parent.from
            to: parent.to
            value: parent.value

            background: Rectangle {
                x: slider.leftPadding
                y: slider.topPadding + slider.availableHeight / 2 - height / 2
                width: slider.availableWidth
                height: 2
                radius: 1
                color: "#2A2A2A"

                Rectangle {
                    width: slider.visualPosition * parent.width
                    height: parent.height
                    color: root.colorBrandSecondary  // Secondary for interactive
                    radius: 1
                }
            }

            handle: Rectangle {
                x: slider.leftPadding + slider.visualPosition * (slider.availableWidth - width)
                y: slider.topPadding + slider.availableHeight / 2 - height / 2
                width: 12
                height: 12
                radius: 6
                color: slider.pressed ? "#FFFFFF" : root.colorBrandSecondary
                border.color: "#FFFFFF"
                border.width: 1

                Behavior on color { ColorAnimation { duration: 150 } }
            }
        }

        Text {
            text: Math.round(slider.value) + " " + parent.unit
            font.pixelSize: 8
            font.weight: Font.Bold
            color: root.colorBrandSecondary  // Secondary for values
            Layout.preferredWidth: 65
        }
    }

    component SmallButton: Rectangle {
        property string label: ""
        property string icon: ""

        Layout.fillWidth: true
        height: 30
        radius: 6
        color: mouse.containsMouse ? root.colorBrandSecondary + "20" : "transparent"  // Hover
        border.color: root.colorBrandSecondary  // Secondary border
        border.width: 1

        Behavior on color { ColorAnimation { duration: 200 } }

        RowLayout {
            anchors.centerIn: parent
            spacing: 5

            Text {
                text: icon
                font.pixelSize: 12
                visible: icon !== ""
            }

            Text {
                text: label
                font.pixelSize: 9
                color: root.colorBrandSecondary  // Secondary text
            }
        }

        MouseArea {
            id: mouse
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: Qt.PointingHandCursor
            onClicked: console.log(label)
        }
    }

    component SettingsDivider: Rectangle {
        Layout.fillWidth: true
        height: 1
        color: root.colorDivider + "40"  // Divider color
    }
}

/*
 ═ ═*═════════════════════════════════════════════════════════════
 ✅ DESIGN SYSTEM SUMMARY:
 ═══════════════════════════════════════════════════════════════

 📝 TYPOGRAPHY HIERARCHY:
 ───────────────────────────────────────────────────────────────
 ✓ Primary Titles    → Bright Cyan (#00FFFF) - Brand, main headers
 ✓ Secondary Titles  → Electric Blue (#0080FF) - Section headers
 ✓ Body Text         → Light Gray (#CCCCCC) - Main content
 ✓ Subtle Text       → Medium Gray (#888888) - Labels, metadata
 ✓ Hint Text         → Dark Gray (#555555) - Placeholders, units

 🎨 BACKGROUNDS:
 ───────────────────────────────────────────────────────────────
 ✓ Cards             → Dark Blue (#161B24)
 ✓ Panels            → Darker (#0D1117)
 ✓ Sections          → Section BG (#1A1F2E)

 🔲 BORDERS & DIVIDERS:
 ───────────────────────────────────────────────────────────────
 ✓ Primary Borders   → Bright Cyan (#00FFFF)
 ✓ Card Borders      → Electric Blue (#0080FF)
 ✓ Dividers          → Deep Teal (#003D4D)

 🖱️ INTERACTIVE ELEMENTS:
 ───────────────────────────────────────────────────────────────
 ✓ Primary Actions   → Mint Green (#00FFA3)
 ✓ Secondary Actions → Electric Blue (#0080FF)
 ✓ Hover States      → Electric Blue @ 30%
 ✓ Active/On States  → Electric Blue

 📊 DATA VISUALIZATION:
 ───────────────────────────────────────────────────────────────
 ✓ Positive/Success  → Mint Green (#00FFA3)
 ✓ Neutral/Info      → Bright Cyan (#00FFFF)
 ✓ Warning           → Electric Blue (#0080FF)
 ✓ Critical/Alert    → Red (#FF4466)

 ═══════════════════════════════════════════════════════════════
 */
