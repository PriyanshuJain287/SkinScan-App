import { Feather, Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { Picker } from '@react-native-picker/picker';
import * as ImagePicker from "expo-image-picker";
import * as Location from 'expo-location';
import { useRouter } from "expo-router";
import React, { useEffect, useState } from "react";


import {
    ActivityIndicator,
    Alert,
    Image,
    ScrollView,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
} from "react-native";

interface WeatherData {
    temp: number;
    uvi: number;
}

// 🌐 CONFIGURE YOUR BACKEND URL HERE
// Use 10.0.2.2 for Android Emulator, localhost for iOS Simulator
const BACKEND_BASE_URL = 'http://192.168.0.103:5000'; // Standardized to Android Emulator default

export default function Home() {
    const [image, setImage] = useState<string | null>(null);
    const router = useRouter();
    const [weather, setWeather] = useState<WeatherData | null>(null);
    const [loadingWeather, setLoadingWeather] = useState(true);
    const [errorMsg, setErrorMsg] = useState<string | null>(null);

    // ✅ CRITICAL FIX: Changed to the actual 'userId' string from your profile.profile-info screenshot
    const USER_ID_PLACEHOLDER = "Pri1"; 
    
    // Clinical State Variables (mapped to 0/1 for the model)
    const [region, setRegion] = useState('trunk'); 
    const [itch, setItch] = useState(0); 
    const [grew, setGrew] = useState(0);
    const [hurt, setHurt] = useState(0);
    const [changed, setChanged] = useState(0);
    const [bleed, setBleed] = useState(0);
    const [elevation, setElevation] = useState(0);

    // Helper for Toggling 0/1 state
    const toggleSymptom = (setter: React.Dispatch<React.SetStateAction<number>>, currentValue: number) => {
        setter(currentValue === 0 ? 1 : 0);
    };

    // ✅ WEATHER FETCH (Using BACKEND_BASE_URL)
    useEffect(() => {
        (async () => {
            let { status } = await Location.requestForegroundPermissionsAsync();
            if (status !== 'granted') {
                setErrorMsg('Location permission denied');
                setLoadingWeather(false);
                return;
            }

            try {
                let location = await Location.getCurrentPositionAsync({
                    accuracy: Location.Accuracy.Highest,
                });

                const { latitude, longitude } = location.coords;

                const BACKEND_WEATHER_URL = `${BACKEND_BASE_URL}/api/weather`;

                const response = await fetch(BACKEND_WEATHER_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        lat: latitude,
                        lon: longitude,
                    }),
                });

                const data = await response.json();

                if (response.ok) {
                    setWeather({ temp: data.temp, uvi: data.uvi });
                } else {
                    setErrorMsg(data.error || 'Failed to fetch weather');
                }
            } catch (error) {
                setErrorMsg('Error fetching weather data from backend.');
                console.error(error);
            } finally {
                setLoadingWeather(false);
            }
        })();
    }, []);

    // ✅ IMAGE PICK (Unchanged)
    const pickImage = async () => {
        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            quality: 1,
        });

        if (!result.canceled) {
            setImage(result.assets[0].uri);
        }
    };

    // ✅ IMAGE UPLOAD (Using BACKEND_BASE_URL and correct USER_ID)
    const uploadImage = async () => {
        if (!image) {
            Alert.alert("No image selected", "Please select an image first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", {
            uri: image,
            name: "photo.jpg",
            type: "image/jpeg",
        } as any);

        // Append all clinical data to the FormData
        formData.append("user_id", USER_ID_PLACEHOLDER); 
        formData.append("region", region);
        formData.append("itch", itch.toString());
        formData.append("grew", grew.toString());
        formData.append("hurt", hurt.toString());
        formData.append("changed", changed.toString());
        formData.append("bleed", bleed.toString());
        formData.append("elevation", elevation.toString());

        try {
            const response = await fetch(`${BACKEND_BASE_URL}/api/upload`, {

                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (response.ok) {
                if (data.status === "No Lesion Detected") {
                    Alert.alert(data.status, data.message);
                } else if (data.status === "Lesion Detected & Classified") {
                    // Get cancer status from backend response
                    const cancerStatus = data.cancer_status || (data.is_cancerous ? "Potentially Cancerous" : "Non-Cancerous");
                    const resultText = `Predicted Class: ${data.predicted_class}
Cancer Status: ${cancerStatus}
Confidence: ${(data.confidence * 100).toFixed(2)}%
Detections: ${data.detections.length} bounding box(es) found.`;
                    Alert.alert("Classification Complete", resultText);
                }
            } else {
                Alert.alert("Error", data.error || "Upload failed");
            }
        } catch (error) {
            console.error("❌ Upload error:", error);
            Alert.alert("Error", "Something went wrong during upload or server connection!");
        }
    };

    // ✅ UV LEVEL LOGIC (Unchanged)
    const getUvIndexInfo = (uvi: number) => {
        const uviValue = Math.round(uvi);
        if (uviValue <= 2) return { level: 'Low', color: '#5cb85c' };
        if (uviValue <= 5) return { level: 'Moderate', color: '#f0ad4e' };
        if (uviValue <= 7) return { level: 'High', color: '#d9534f' };
        if (uviValue <= 10) return { level: 'Very High', color: '#d9534f' };
        return { level: 'Extreme', color: '#8B008B' };
    };
    
    // Helper Component for Clinical Inputs
    const ClinicalInput = ({ label, value, setter }: { label: string, value: number, setter: React.Dispatch<React.SetStateAction<number>> }) => (
        <View style={styles.clinicalInputRow}>
            <Text style={styles.clinicalLabel}>{label}:</Text>
            <TouchableOpacity
                style={[
                    styles.clinicalSwitch,
                    { backgroundColor: value === 1 ? '#4b7bec' : '#ddd' },
                ]}
                onPress={() => toggleSymptom(setter, value)}
            >
                <Text style={styles.clinicalSwitchText}>{value === 1 ? 'Yes' : 'No'}</Text>
            </TouchableOpacity>
        </View>
    );

    return (
        <ScrollView contentContainerStyle={styles.container}>

            {/* HEADER */}
            <View style={styles.header}>
                <Text style={styles.title}>Skin Health Analysis</Text>
                <Text style={styles.subtitle}>
                    Upload a photo of a skin lesion for an AI-powered analysis.
                </Text>
            </View>

            {/* IMAGE UPLOAD CARD */}
            <View style={styles.card}>
                <Text style={styles.cardTitle}>Image Upload</Text>
                <TouchableOpacity onPress={pickImage} style={styles.imagePicker}>
                    {image ? (
                        <Image source={{ uri: image }} style={styles.image} />
                    ) : (
                        <View style={styles.placeholder}>
                            <MaterialCommunityIcons name="image-plus" size={50} color="#a0a0a0" />
                            <Text style={styles.placeholderText}>Tap to select an image</Text>
                        </View>
                    )}
                </TouchableOpacity>
            </View>
            
            {/* CLINICAL INPUT CARD (NEW) */}
            <View style={styles.card}>
                <Text style={styles.cardTitle}>Clinical Information</Text>
                <Text style={styles.cardSubtitle}>Describe the lesion characteristics needed for the AI.</Text>
                
                {/* REGION INPUT */}
                <View style={styles.clinicalInputRow}>
                    <Text style={styles.clinicalLabel}>Region:</Text>
                    <View style={styles.pickerContainer}>
                        <Picker
                            selectedValue={region}
                            style={styles.picker}
                            onValueChange={(itemValue) => setRegion(itemValue)}
                        >
                            <Picker.Item label="Trunk" value="trunk" />
                            <Picker.Item label="Face" value="face" />
                            <Picker.Item label="Lower Extremity" value="lower_extremity" />
                            <Picker.Item label="Upper Extremity" value="upper_extremity" />
                            <Picker.Item label="Head/Neck" value="head_neck" />
                            <Picker.Item label="Genital" value="genital" />
                            <Picker.Item label="Palms/Soles" value="palms_soles" />
                        </Picker>
                    </View>
                </View>

                {/* Symptom Inputs (0/1) */}
                <ClinicalInput label="Itches (Pruritus)" value={itch} setter={setItch} />
                <ClinicalInput label="Has Grown" value={grew} setter={setGrew} />
                <ClinicalInput label="Hurts (Pain)" value={hurt} setter={setHurt} />
                <ClinicalInput label="Has Changed Shape/Color" value={changed} setter={setChanged} />
                <ClinicalInput label="Bleeds" value={bleed} setter={setBleed} />
                <ClinicalInput label="Is Elevated" value={elevation} setter={setElevation} />

                {image && (
                    <TouchableOpacity style={styles.uploadButton} onPress={uploadImage}>
                        <MaterialCommunityIcons name="cloud-upload-outline" size={24} color="white" />
                        <Text style={styles.buttonText}>Upload & Analyze</Text>
                    </TouchableOpacity>
                )}
            </View>

            {/* WEATHER CARD */}
            <View style={styles.card}>
                <Text style={styles.cardTitle}>Today's Local Conditions</Text>
                <Text style={styles.cardSubtitle}>Stay informed about your environment.</Text>

                {loadingWeather ? (
                    <ActivityIndicator size="large" color="#4b7bec" />
                ) : errorMsg ? (
                    <Text style={styles.errorText}>{errorMsg}</Text>
                ) : weather ? (
                    <View style={styles.weatherContainer}>
                        <View style={styles.weatherItem}>
                            <Feather name="thermometer" size={24} color="#4b7bec" />
                            <Text style={styles.weatherData}>{Math.round(weather.temp)}°C</Text>
                            <Text style={styles.weatherLabel}>Temperature</Text>
                        </View>

                        <View style={styles.weatherItem}>
                            <Feather name="sun" size={24} color={getUvIndexInfo(weather.uvi).color} />
                            <Text style={[styles.weatherData, { color: getUvIndexInfo(weather.uvi).color }]}>
                                {getUvIndexInfo(weather.uvi).level}
                            </Text>
                            <Text style={styles.weatherLabel}>UV Index</Text>
                        </View>
                    </View>
                ) : null}
            </View>

            {/* NAVIGATION BUTTONS */}
            <TouchableOpacity style={styles.blogsButton} onPress={() => router.push('/blog')}>
                <MaterialCommunityIcons name="book-open-page-variant" size={22} color="#fff" />
                <Text style={styles.blogsButtonText}>View Blogs</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.hospitalsButton} onPress={() => router.push('/hospitals')}>
                <Feather name="map-pin" size={22} color="#fff" />
                <Text style={styles.hospitalsButtonText}>Find Hospitals</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.remindersButton} onPress={() => router.push('/reminders')}>
                <MaterialCommunityIcons name="bell-outline" size={22} color="#fff" />
                <Text style={styles.remindersButtonText}>Set Reminders</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.profileButton} onPress={() => router.push('/profile')}>
                <MaterialCommunityIcons name="account-edit-outline" size={22} color="#fff" />
                <Text style={styles.profileButtonText}>Edit Profile</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.skinTypeButton} onPress={() => router.push('/skinTypeQuiz')}>
                <MaterialCommunityIcons name="lotion" size={22} color="#fff" />
                <Text style={styles.skinTypeButtonText}>Know Your Skin Type</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.chatbotButton} onPress={() => router.push('/chatbot')}>
                <Ionicons name="chatbubble-ellipses" size={22} color="#fff" />
                <Text style={styles.chatbotButtonText}>AI Assistant</Text>
            </TouchableOpacity>

        </ScrollView>
    );
}

/* STYLES (Unchanged) */
const styles = StyleSheet.create({
    container: { flexGrow: 1, padding: 16, backgroundColor: '#f5f5f5', alignItems: 'center' },
    header: { width: '100%', marginBottom: 20, alignItems: 'center' },
    title: { fontSize: 26, fontWeight: "bold", color: '#333' },
    subtitle: { fontSize: 16, color: '#666', textAlign: 'center', marginTop: 4 },
    card: { backgroundColor: '#ffffff', borderRadius: 12, padding: 20, width: '100%', marginBottom: 20, alignItems: 'center' },
    imagePicker: { width: '100%', height: 200, borderRadius: 12, justifyContent: 'center', alignItems: 'center', backgroundColor: '#f0f0f0', borderWidth: 2, borderColor: '#ddd', borderStyle: 'dashed', marginBottom: 10 },
    image: { width: '100%', height: '100%', borderRadius: 10 },
    placeholder: { alignItems: 'center' },
    placeholderText: { marginTop: 8, color: '#a0a0a0', fontSize: 16 },
    uploadButton: { 
        flexDirection: 'row', alignItems: 'center', backgroundColor: '#4b7bec', 
        paddingVertical: 12, paddingHorizontal: 30, borderRadius: 25, 
        width: '100%', justifyContent: 'center', marginTop: 20 
    },
    buttonText: { color: '#fff', fontSize: 18, fontWeight: 'bold', marginLeft: 10 },

    cardTitle: { fontSize: 20, fontWeight: 'bold', color: '#333', marginBottom: 5 },
    cardSubtitle: { fontSize: 14, color: '#666', textAlign: 'center', marginBottom: 15 },

    // New Clinical Styles
    clinicalInputRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        width: '100%',
        paddingVertical: 8,
        borderBottomWidth: 1,
        borderBottomColor: '#eee',
    },
    clinicalLabel: {
        fontSize: 16,
        color: '#333',
        fontWeight: '500',
        flex: 2,
    },
    clinicalSwitch: {
        paddingHorizontal: 15,
        paddingVertical: 5,
        borderRadius: 15,
        minWidth: 70,
        alignItems: 'center',
    },
    clinicalSwitchText: {
        color: '#fff',
        fontWeight: 'bold',
    },
    pickerContainer: {
        flex: 1,
        borderWidth: 1,
        borderColor: '#ddd',
        borderRadius: 8,
        overflow: 'hidden',
        height: 40,
        justifyContent: 'center',
    },
    picker: {
        width: '100%',
        height: 40,
    },

    weatherContainer: { flexDirection: 'row', justifyContent: 'space-around', width: '100%' },
    weatherItem: { alignItems: 'center', flex: 1 },
    weatherData: { fontSize: 22, fontWeight: 'bold', marginTop: 8 },
    weatherLabel: { fontSize: 14, color: '#666' },

    errorText: { fontSize: 16, color: '#d9534f', marginVertical: 20, textAlign: 'center' },

    blogsButton: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#4b7bec', padding: 12, borderRadius: 25, marginBottom: 10, width: '100%', justifyContent: 'center' },
    blogsButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold', marginLeft: 8 },

    hospitalsButton: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#e74c3c', padding: 12, borderRadius: 25, marginBottom: 10, width: '100%', justifyContent: 'center' },
    hospitalsButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold', marginLeft: 8 },

    remindersButton: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#ff6b6b', padding: 12, borderRadius: 25, marginBottom: 10, width: '100%', justifyContent: 'center' },
    remindersButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold', marginLeft: 8 },

    profileButton: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#9b59b6', padding: 12, borderRadius: 25, marginBottom: 10, width: '100%', justifyContent: 'center' },
    profileButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold', marginLeft: 8 },
    chatbotButton: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    backgroundColor: '#2ecc71', 
    padding: 12, 
    borderRadius: 25, 
    marginBottom: 20,
    width: '100%', 
    justifyContent: 'center' 
    },
    skinTypeButton: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    backgroundColor: '#FF6B35', 
    padding: 12, 
    borderRadius: 25, 
    marginBottom: 10, 
    width: '100%', 
    justifyContent: 'center' 
},
skinTypeButtonText: { 
    color: '#fff', 
    fontSize: 16, 
    fontWeight: 'bold', 
    marginLeft: 8 
},
    chatbotButtonText: { 
    color: '#fff', 
    fontSize: 16, 
    fontWeight: 'bold', 
    marginLeft: 8 
    },
});
