import { useRouter } from "expo-router";
import React, { useState } from "react";
import { ActivityIndicator, Alert, Dimensions, StyleSheet, Text, TextInput, TouchableOpacity, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useProfile } from "./context/ProfileContext";

export default function Login() {
  const safeAreaInsets = useSafeAreaInsets();
  const router = useRouter();
  const screenHeight = Dimensions.get("window").height;
  const { setUserCredentials } = useProfile();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert("Error", "Please fill in all fields");
      return;
    }

    try {
      setLoading(true);
      const response = await fetch("http://192.168.0.103:5000/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (response.ok) {
        console.log("Login successful, setting user credentials");
        
        // Extract username from email or use full_name as username
        const username = data.full_name || email.split('@')[0];
        const userEmail = email;
        
        // Set credentials in profile context
        setUserCredentials(username, userEmail);
        
        Alert.alert("Success", data.message || "Login successful!");
        router.push("/home"); // Navigate to home or dashboard page
      } else {
        Alert.alert("Error", data.message || "Invalid credentials");
      }
    } catch (error) {
      console.error(error);
      Alert.alert("Error", "Unable to connect to server. Please check your connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <View
      style={[
        styles.container,
        {
          paddingTop: safeAreaInsets.top + screenHeight * 0.08,
          paddingBottom: safeAreaInsets.bottom + screenHeight * 0.05,
        },
      ]}
    >
      <Text style={styles.title}>Login</Text>
      <Text style={styles.subtitle}>Welcome back to SkinScan</Text>

      {/* Email */}
      <Text style={styles.label}>Email</Text>
      <TextInput
        style={styles.input}
        placeholder="Enter your email"
        placeholderTextColor="#778ca3"
        keyboardType="email-address"
        value={email}
        onChangeText={setEmail}
        autoCapitalize="none"
        autoComplete="email"
      />

      {/* Password */}
      <Text style={styles.label}>Password</Text>
      <TextInput
        style={styles.input}
        placeholder="Enter your password"
        placeholderTextColor="#778ca3"
        secureTextEntry
        value={password}
        onChangeText={setPassword}
        autoCapitalize="none"
      />

      <TouchableOpacity 
        style={[styles.button, loading && styles.disabledButton]} 
        onPress={handleLogin}
        disabled={loading}
      >
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Login</Text>
        )}
      </TouchableOpacity>

      <TouchableOpacity onPress={() => router.push("/signup")}>
        <Text style={styles.logs}>Don't have an account? Sign Up</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: "center", backgroundColor: "#F5FCFF", paddingHorizontal: 20 },
  label: { width: "100%", fontSize: 14, fontWeight: "500", color: "#34495e", marginBottom: 5 },
  title: { fontSize: 36, fontWeight: "bold", color: "#4b7bec", marginBottom: 5 },
  subtitle: { fontSize: 16, color: "#778ca3", marginBottom: 20 },
  input: { 
    width: "100%", 
    borderWidth: 1, 
    borderColor: "#d1d8e0", 
    borderRadius: 10, 
    padding: 15, 
    fontSize: 16, 
    marginBottom: 15, 
    color: "#2d3436",
    backgroundColor: "#fff"
  },
  button: { 
    backgroundColor: "#4b7bec", 
    paddingVertical: 15, 
    paddingHorizontal: 40, 
    borderRadius: 25, 
    marginTop: 10,
    width: "100%",
    alignItems: "center"
  },
  disabledButton: {
    backgroundColor: "#a5b1c2",
  },
  buttonText: { color: "#fff", fontSize: 18, fontWeight: "bold" },
  logs: { fontSize: 16, color:'#4b7bec', textAlign: 'center', marginTop: 20 },
});
