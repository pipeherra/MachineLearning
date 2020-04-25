clear all, close all

Daten = load ('Treppensteigen_Unterarm_proband61');

    Timestamp = Daten(:,2); % Zeit
    accelX = Daten(:,3); %(m/s^2)
    accelY = Daten(:,4); %(m/s^2)
    accelZ = Daten(:,5); %(m/s^2)
    gyroX = Daten(:,6); %(rad/s)
    gyroY = Daten(:,7); %(rad/s)
    gyroZ = Daten(:,8); %(rad/s)
    magnetX = Daten(:,9); %(uT)
    magnetY = Daten(:,10); %(uT)
    magnetZ = Daten(:,11); %(uT)
    
    magnetZmean =mean(accelZ);
    magnetYmean =mean(accelY);
    magnetXmean =mean(accelX);
    
    std(accelZ)
    figure()
    plot((Timestamp-Timestamp(1))./1000,((accelZ-magnetZmean).^2))
    figure()
    plot((Timestamp-Timestamp(1))./1000,((accelY-magnetYmean).^2))
    figure()
    plot((Timestamp-Timestamp(1))./1000,((accelX-magnetXmean).^2))
    
 
    
   