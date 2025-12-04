import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Alert, ScrollView, StyleSheet, Text, TouchableOpacity } from 'react-native';

// --- Quiz Data ---
const questions = [
  {
    question: "How does your skin feel after washing your face and patting it dry?",
    options: [
      { text: "Tight and slightly uncomfortable.", type: 'a' },
      { text: "Smooth and soft.", type: 'b' },
      { text: "Still a bit oily or shiny in some areas.", type: 'c' },
      { text: "Itchy or red in some spots.", type: 'd' },
      { text: "Oily in the T-zone but dry on the cheeks.", type: 'e' },
    ],
  },
  {
    question: "How would you describe your pores?",
    options: [
      { text: "Small and not very noticeable.", type: 'a' },
      { text: "Visible but not overly large.", type: 'b' },
      { text: "Large and easily visible.", type: 'c' },
      { text: "They can be red or inflamed.", type: 'd' },
      { text: "Larger in the T-zone and smaller on the cheeks.", type: 'e' },
    ],
  },
  {
    question: "How often do you experience breakouts (pimples, blackheads)?",
    options: [
      { text: "Rarely, if ever.", type: 'a' },
      { text: "Occasionally, usually around hormonal changes.", type: 'b' },
      { text: "Frequently.", type: 'c' },
      { text: "I get red bumps and irritation, but not always pimples.", type: 'd' },
      { text: "Mostly in the T-zone.", type: 'e' },
    ],
  },
  {
    question: "How does your skin look by midday without any makeup?",
    options: [
      { text: "Flaky or with dry patches.", type: 'a' },
      { text: "Fairly even and balanced.", type: 'b' },
      { text: "Shiny all over.", type: 'c' },
      { text: "Red or blotchy.", type: 'd' },
      { text: "Shiny on the forehead and nose, but matte on the cheeks.", type: 'e' },
    ],
  },
];

const results = {
  a: { type: "Dry Skin", description: "Your skin produces less sebum, feeling tight and possibly flaky. Focus on hydration with ingredients like hyaluronic acid." },
  b: { type: "Normal Skin", description: "Your skin is well-balanced. Maintain a consistent routine with a gentle cleanser, moisturizer, and sunscreen." },
  c: { type: "Oily Skin", description: "Your skin produces excess sebum, leading to shine and breakouts. Use oil-free products and ingredients like salicylic acid." },
  d: { type: "Sensitive Skin", description: "Your skin is easily irritated. Use gentle, fragrance-free products and always patch-test new items." },
  e: { type: "Combination Skin", description: "You have both oily (T-zone) and dry/normal (cheeks) areas. Use a gentle cleanser and a lightweight moisturizer." },
};

export default function SkinTypeQuiz() {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<string[]>([]);
  const router = useRouter();

  const handleAnswer = (answerType: string) => {
    const nextAnswers = [...answers];
    nextAnswers[currentQuestionIndex] = answerType; // save answer for this question
    setAnswers(nextAnswers);

    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      showResult(nextAnswers);
    }
  };

  const handleGoBack = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const showResult = (finalAnswers: string[]) => {
    const counts: { [key: string]: number } = {};
    for (const ans of finalAnswers) {
      counts[ans] = (counts[ans] || 0) + 1;
    }

    const resultType = Object.keys(counts).reduce((a, b) => (counts[a] > counts[b] ? a : b));
    const resultData = results[resultType as keyof typeof results];

    Alert.alert(
      `Your Skin Type: ${resultData.type}`,
      resultData.description,
      [{ text: 'OK', onPress: () => router.back() }]
    );
  };

  const currentQuestion = questions[currentQuestionIndex];

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.questionText}>{currentQuestion.question}</Text>
      
      {currentQuestion.options.map((option, index) => (
        <TouchableOpacity
          key={index}
          style={styles.optionButton}
          onPress={() => handleAnswer(option.type)}
        >
          <Text style={styles.optionText}>{option.text}</Text>
        </TouchableOpacity>
      ))}

      {/* Go Back Button */}
      {currentQuestionIndex > 0 && (
        <TouchableOpacity style={styles.goBackButton} onPress={handleGoBack}>
          <Text style={styles.goBackText}>Go Back</Text>
        </TouchableOpacity>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#F5FCFF',
    justifyContent: 'center',
  },
  questionText: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#4b7bec',
    marginBottom: 30,
    textAlign: 'center',
  },
  optionButton: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#d1d8e0',
    marginBottom: 15,
  },
  optionText: {
    fontSize: 16,
    color: '#2d3436',
  },
  goBackButton: {
    marginTop: 20,
    padding: 12,
    backgroundColor: '#4b7bec',
    borderRadius: 10,
    alignItems: 'center',
  },
  goBackText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
