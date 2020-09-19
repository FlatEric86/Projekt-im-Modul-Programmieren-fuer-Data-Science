# PDS_Project |Advektion- und Diffusionsbasiertes Stoffausbreitungsmodel auf Basis eines künstlichen neuronalen Netzes

Dieses Projekt ist die Modulabschlussarbeit im Modul "Programmieren für Data Science" aus dem ersten Semester meines Master-Studiums "Data Science".
Es behandelt den Versuch eines einfachen Modells zur Modellierung einfacher Stoffausbreitung in porösen Grundwasserleitern auf Basis eines künstlichen neuronalen Netzes.
Die Projektbeschreibung ist in der Datei <semesterarbeit_pds_alexander_prinz.pdf> zu finden.
Zudem ist in der Datei <Prsnt_v_1.pdf> eine Zusammenfassung des Konzeptes in Form einer Präsentation zu finden.
Es konnte gezeigt werden, dass ein recht einfaches neuronales Netz in der Lage ist, ein solches Problem zu lösen.
Zwar ist das neuronale Netz innerhalb dieses Projekts nur in der Lage stationäre Probleme zu lösen (Modellausgabe zu einem bestimmten Zeitpunkt an einem bestimmten Ort), jedoch könnte eine Erweiterung des Modells etwa durch mehr Tiefe und auf Basis von Faltung (CNN) oder Rekursion (RNN) bzw. hybridformen beider vermutlich auch instationäre Probleme lösen.
Der Vorteil eines solchen Netzes gegenüber der konventionellen Methoden über Einschrittverfahren wäre zumindest bei reinen CNN der, dass jeder i'te Zeitschritt sofort berechntet werden kann, also nicht die Zeitschritte ab dem Anfangswert bis zum i-1'ten Schritt bekannt sein müssen.
Dagegen würden RNN keinen wirklichen Mehrgewinn in Pucto Laufzeit bieten, da sich die Ergebnisse aus den vorhergehenden ableiten, wie es eben auch bei den konventionellen Verfahren nötig ist. Jedoch können neuronale Netze bestimmte Kopplungseffekte verschiedenen physikalischer und auch chemischer Prozesse, welche konventionell möglicherweise nicht modellierbar sind, erlernen und modellieren.


Die Projektarbeit wurde vom Dozenten mit einer 1.0 bewertet.
