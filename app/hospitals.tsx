import { Feather } from '@expo/vector-icons';
import * as Location from 'expo-location';
import React, { useEffect, useState } from 'react';
import {
    ActivityIndicator,
    Alert,
    Linking,
    ScrollView,
    StyleSheet,
    Text,
    TouchableOpacity,
    View
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

interface Hospital {
  name: string;
  address: string;
  rating?: number;
  phone?: string;
  website?: string;
  location: {
    lat: number;
    lng: number;
  };
  open_now?: boolean;
}

export default function HospitalsScreen() {
  const safeAreaInsets = useSafeAreaInsets();
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [location, setLocation] = useState<{ lat: number; lon: number } | null>(null);

  useEffect(() => {
    fetchLocationAndHospitals();
  }, []);

  const fetchLocationAndHospitals = async () => {
    try {
      // Request location permission
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setError('Location permission denied');
        setLoading(false);
        return;
      }

      // Get current location
      let location = await Location.getCurrentPositionAsync({});
      const { latitude, longitude } = location.coords;
      setLocation({ lat: latitude, lon: longitude });

      // Fetch nearby hospitals
      await fetchNearbyHospitals(latitude, longitude);
    } catch (err) {
      setError('Error getting location');
      setLoading(false);
    }
  };

  const fetchNearbyHospitals = async (lat: number, lon: number) => {
    try {
      const response = await fetch('http://192.168.0.103:5000/api/nearby-hospitals', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lat: lat,
          lon: lon,
          radius: 10000, // 10km radius
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setHospitals(data.hospitals || []);
      } else {
        setError(data.error || 'Failed to fetch hospitals');
      }
    } catch (err) {
      setError('Unable to connect to server');
    } finally {
      setLoading(false);
    }
  };

  const openMaps = (lat: number, lng: number, name: string) => {
    const url = `https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}&travelmode=driving&dir_action=navigate`;
    Linking.openURL(url).catch(err =>
      Alert.alert('Error', 'Could not open maps app')
    );
  };

  const makePhoneCall = (phone: string) => {
    Linking.openURL(`tel:${phone}`).catch(err =>
      Alert.alert('Error', 'Could not make phone call')
    );
  };

  const openWebsite = (website: string) => {
    Linking.openURL(website).catch(err =>
      Alert.alert('Error', 'Could not open website')
    );
  };

  const refreshHospitals = () => {
    setLoading(true);
    setError(null);
    if (location) {
      fetchNearbyHospitals(location.lat, location.lon);
    } else {
      fetchLocationAndHospitals();
    }
  };

  const renderHospitalCard = (hospital: Hospital, index: number) => (
    <View key={index} style={styles.hospitalCard}>
      <View style={styles.hospitalHeader}>
        <Text style={styles.hospitalName}>{hospital.name}</Text>
        {hospital.rating && (
          <View style={styles.ratingContainer}>
            <Feather name="star" size={16} color="#FFD700" />
            <Text style={styles.rating}>{hospital.rating}</Text>
          </View>
        )}
      </View>

      <View style={styles.hospitalInfo}>
        <Feather name="map-pin" size={16} color="#666" />
        <Text style={styles.hospitalAddress}>{hospital.address}</Text>
      </View>

      {hospital.open_now !== undefined && (
        <View style={styles.hospitalInfo}>
          <Feather 
            name="clock" 
            size={16} 
            color={hospital.open_now ? '#4CAF50' : '#F44336'} 
          />
          <Text style={[
            styles.openStatus,
            { color: hospital.open_now ? '#4CAF50' : '#F44336' }
          ]}>
            {hospital.open_now ? 'Open Now' : 'Closed'}
          </Text>
        </View>
      )}

      <View style={styles.actionButtons}>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => openMaps(hospital.location.lat, hospital.location.lng, hospital.name)}
        >
          <Feather name="navigation" size={18} color="#4b7bec" />
          <Text style={styles.actionButtonText}>Directions</Text>
        </TouchableOpacity>

        {hospital.phone && (
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => makePhoneCall(hospital.phone!)}
          >
            <Feather name="phone" size={18} color="#4b7bec" />
            <Text style={styles.actionButtonText}>Call</Text>
          </TouchableOpacity>
        )}

        {hospital.website && (
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => openWebsite(hospital.website!)}
          >
            <Feather name="globe" size={18} color="#4b7bec" />
            <Text style={styles.actionButtonText}>Website</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );

  return (
    <View style={[styles.container, { paddingTop: safeAreaInsets.top }]}>
      <View style={styles.header}>
        <Text style={styles.title}>Nearby Cancer Hospitals</Text>
        <Text style={styles.subtitle}>
          Specialized oncology centers near your location
        </Text>
      </View>

      <TouchableOpacity style={styles.refreshButton} onPress={refreshHospitals}>
        <Feather name="refresh-cw" size={18} color="#4b7bec" />
        <Text style={styles.refreshButtonText}>Refresh</Text>
      </TouchableOpacity>

      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4b7bec" />
          <Text style={styles.loadingText}>Finding nearby hospitals...</Text>
        </View>
      ) : error ? (
        <View style={styles.errorContainer}>
          <Feather name="alert-triangle" size={50} color="#F44336" />
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={refreshHospitals}>
            <Text style={styles.retryButtonText}>Try Again</Text>
          </TouchableOpacity>
        </View>
      ) : hospitals.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Feather name="search" size={50} color="#666" />
          <Text style={styles.emptyText}>No hospitals found in your area</Text>
          <Text style={styles.emptySubtext}>Try increasing the search radius</Text>
        </View>
      ) : (
        <ScrollView 
          style={styles.hospitalsList}
          showsVerticalScrollIndicator={false}
        >
          {hospitals.map(renderHospitalCard)}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    paddingHorizontal: 16,
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginTop: 8,
  },
  refreshButton: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'center',
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  refreshButtonText: {
    color: '#4b7bec',
    fontWeight: '600',
    marginLeft: 8,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  errorText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginTop: 16,
    marginBottom: 24,
  },
  retryButton: {
    backgroundColor: '#4b7bec',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 25,
  },
  retryButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  emptyText: {
    fontSize: 18,
    color: '#666',
    textAlign: 'center',
    marginTop: 16,
    fontWeight: '600',
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
    marginTop: 8,
  },
  hospitalsList: {
    flex: 1,
  },
  hospitalCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  hospitalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  hospitalName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    flex: 1,
    marginRight: 8,
  },
  ratingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF9C4',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  rating: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginLeft: 4,
  },
  hospitalInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  hospitalAddress: {
    fontSize: 14,
    color: '#666',
    marginLeft: 8,
    flex: 1,
  },
  openStatus: {
    fontSize: 14,
    fontWeight: '500',
    marginLeft: 8,
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#f8f9fa',
  },
  actionButtonText: {
    color: '#4b7bec',
    fontWeight: '600',
    marginLeft: 6,
    fontSize: 14,
  },
});
