import { Stack } from "expo-router";
import { ProfileProvider } from "./context/ProfileContext";
import ChatbotScreen from './chatbot';

export default function RootLayout() {
  return (
    <ProfileProvider>
      <Stack>
        <Stack.Screen name="index" options={{ title: "Welcome" }} />
        <Stack.Screen name="signup" options={{ title: "Signup" }} />
        <Stack.Screen name="login" options={{ title: "Login" }} />
        <Stack.Screen name="home" options={{ title: "Home" }} />
        <Stack.Screen name="skinTypeQuiz" options={{ title: "Know Your Skin Type" }} />
        <Stack.Screen name="profile" options={{ title: "Your Profile" }} />
        <Stack.Screen name="chatbot" options={{ title: "SkinAI Assistant" }} />
      </Stack>
    </ProfileProvider>
  );
}
