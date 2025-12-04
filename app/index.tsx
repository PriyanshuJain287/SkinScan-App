import { useRouter } from 'expo-router';
import React from 'react';
import { Dimensions, Image, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

export default function LandingPage() {
  const safeAreaInsets = useSafeAreaInsets();
  const router = useRouter();

  const screenHeight = Dimensions.get("window").height;

  return (
    <View
      style={[
        styles.container,
        {
          paddingTop: safeAreaInsets.top + screenHeight * 0.06,   // 6% of screen height
          paddingBottom: safeAreaInsets.bottom + screenHeight * 0.04, // 4% of screen height
        },
      ]}
    >
      <Text style={styles.title}>SkinScan</Text>
      <Text style={styles.subtitle}>Smarter Skin Health, Powered by AI</Text>

      <Image
        source={{ uri: 'https://cdn-icons-png.flaticon.com/512/2762/2762765.png' }}
        style={styles.image}
      />

      <Text style={styles.description}>
        Fast, simple, and designed to keep your skin safe.
      </Text>

      {/* Sign Up Button */}
      <TouchableOpacity
        style={styles.button}
        onPress={() => router.push('/signup')} // Navigate to Sign Up
      >
        <Text style={styles.buttonText}>Join Us!</Text>
      </TouchableOpacity>

      {/* Start Scan Button */}

      {/* Login link */}
      <TouchableOpacity onPress={() => router.push("/login")}>
        <Text style={styles.logs}>
          Already a member? Login
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { 
    flex: 1, 
    alignItems: 'center', 
    backgroundColor: '#F5FCFF', 
    paddingHorizontal: 20 
  },
  title: { fontSize: 36, fontWeight: 'bold', color: '#4b7bec', marginBottom: 5 },
  subtitle: { fontSize: 16, color: '#778ca3', marginBottom: 20 },
  image: { width: 200, height: 200, marginVertical: 20 },
  description: { fontSize: 16, textAlign: 'center', color: '#2d3436', marginBottom: 18, marginTop: 10, fontWeight:"500"},
  logs: { fontSize: 16, color:'#4b7bec', textAlign: 'center', marginTop: 20 },
  button: { backgroundColor: '#4b7bec', paddingVertical: 15, paddingHorizontal: 40, borderRadius: 25 },
  buttonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
});
