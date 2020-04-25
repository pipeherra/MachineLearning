clear all, close all

% Eingaben 

CSV_filename = input('Datensatz (z.B. test4): ','s');
CSV_Daten = load (CSV_filename);
%CSV_Daten = importdata(CSV_filename, ' ', 1);
%Anzahl_Sensor = input('Anzahl der Sensoren (z.B. 5): ','s');
Anzahl_Sensor = 5;
Probant = input('Bezeichnung Probant (z.B. XYZ): ','s');
Aktion = input('Aktion (z.B. Treppensteigen): ','s');

% Linker U.Schenkel ID 01324188
% Linker O.Schenkel ID 01324189
% Linker U.Arm      ID 01324180
% Linker O.Arm      ID 01324185
% Rücken            ID 01324181

FIX_IDs = [01324188,01324189,01324180,01324185,01324181];
ID_NAMES = {'Unterschenkel','Oberschenkel','Unterarm','Oberarm','Ruecken'};

ID = CSV_Daten(:,1);
Sensor = cell(1,Anzahl_Sensor);

for n = 1:Anzahl_Sensor
    Sensor{n} = CSV_Daten(n,1);
end

Daten = cell(1,Anzahl_Sensor);

for n = 1:Anzahl_Sensor
    Daten{n} = CSV_Daten(ID == Sensor{n},:);
end

for n = 1:Anzahl_Sensor
    
    ID = Daten{n}(:,1);
    
    Dateiname = [Aktion '_' ID_NAMES{find(FIX_IDs == ID(n,1))} '_' Probant];
    saveDaten = fopen (Dateiname, 'w');
    %fprintf ( saveDaten , 'ID,Timestamp,accelX (m/s^2),accelY (m/s^2),accelZ (m/s^2),gyroX (rad/s),gyroY (rad/s),gyroZ (rad/s),magnetX (uT),magnetY (uT),magnetZ (uT)\n');
    
    figure(n)
    
    Timestamp = Daten{n}(:,2); % Zeit
    accelX = Daten{n}(:,3); %(m/s^2)
    accelY = Daten{n}(:,4); %(m/s^2)
    accelZ = Daten{n}(:,5); %(m/s^2)
    gyroX = Daten{n}(:,6); %(rad/s)
    gyroY = Daten{n}(:,7); %(rad/s)
    gyroZ = Daten{n}(:,8); %(rad/s)
    magnetX = Daten{n}(:,9); %(uT)
    magnetY = Daten{n}(:,10); %(uT)
    magnetZ = Daten{n}(:,11); %(uT)

    for i = 1:3
        
    subplot(3,1,i)
    
        if i == 1
            plot((Timestamp-Timestamp(1))./1000,accelX);
            hold on
            plot((Timestamp-Timestamp(1))./1000,accelY,'g');
            plot((Timestamp-Timestamp(1))./1000,accelZ,'r');
            hold off
            grid on
            title(ID_NAMES{find(FIX_IDs == ID(n,1))});
            ylabel('m/s^2 \rightarrow')
            xlabel('Zeit / s \rightarrow')
            legend('accelX','accelY','accelZ')
        end
        
        if i==2
            plot((Timestamp-Timestamp(1))./1000,gyroX);
            hold on
            plot((Timestamp-Timestamp(1))./1000,gyroY,'g');
            plot((Timestamp-Timestamp(1))./1000,gyroZ,'r');
            hold off
            grid on
            title(ID_NAMES{find(FIX_IDs == ID(n,1))});
            ylabel('rad/s \rightarrow')
            xlabel('Zeit / s \rightarrow')
            legend('gyroX','gyroY','gyroZ')
        end
        
        if i==3
            plot((Timestamp-Timestamp(1))./1000,magnetX);
            hold on
            plot((Timestamp-Timestamp(1))./1000,magnetY,'g');
            plot((Timestamp-Timestamp(1))./1000,magnetZ,'r');
            hold off
            grid on
            title(ID_NAMES{find(FIX_IDs == ID(n,1))});
            ylabel('\mu T \rightarrow')
            xlabel('Zeit / s \rightarrow')
            legend('magnetX','magnetY','magnetZ')
        end
        
    end
       fprintf ( saveDaten , '%d,%d,%3.6f,%3.6f,%3.6f,%3.6f,%3.6f,%3.6f,%3.6f,%3.6f,%3.6f\n',[ID';Timestamp'; accelX'; accelY'; accelZ';gyroX';gyroY';gyroZ';magnetX';magnetY';magnetZ']);
        %Dateiname = [Aktion '_' ID_NAMES{find(FIX_IDs == ID(n,1))} '_' Probant '.csv']        
        fclose ( saveDaten );
        Dateiname
end

%  Datenspeichern als <Bewegungsablauf>_ProbantXY.csv


%saveDaten = fopen (Dateiname, 'w');
%fprintf ( saveDaten , 'Jahr Kapital Zinsen Tilgung \n');
%fprintf ( saveDaten , '%4d: %9.2 f %7.2 f %7.2 f\n', Plan ');
%fclose ( saveDaten );
