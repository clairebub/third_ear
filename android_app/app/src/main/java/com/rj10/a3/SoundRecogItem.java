package com.rj10.a3;

import java.util.Date;

public class SoundRecogItem {
    public Date timestamp;
    public String textRecognized;
    public String wavFileRecorded;
    public String wavSoundType;

    public static SoundRecogItem createItemForTextRecognized(Date timestamp, String textRecognized) {
        SoundRecogItem item = new SoundRecogItem(timestamp);
        item.textRecognized = textRecognized;
        return item;
    }

    public static SoundRecogItem createItemForWavFileRecorded(Date timestamp, String wavFileRecorded, String soundClass) {
        SoundRecogItem item = new SoundRecogItem(timestamp);
        item.wavFileRecorded = wavFileRecorded;
        item.wavSoundType = soundClass;
        return item;
    }

    private SoundRecogItem(Date timestamp) {
        this.timestamp = timestamp;
    }

}
