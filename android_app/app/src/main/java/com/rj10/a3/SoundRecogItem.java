package com.rj10.a3;

import java.util.Date;

public class SoundRecogItem {
    public String mStatus;
    public String mLabel;
    private String mWavFileName;
    private Date mTimestamp;


    public SoundRecogItem(String status, Date timestamp) {
        mStatus = status;
        mTimestamp = timestamp;
    }

    public String getWavFileName() {
        return mWavFileName;
    }

    public void setWavFileName(String fileName) {
        mWavFileName = fileName;
    }

    public String getLabel() {
        return mLabel;
    }

    public Date getTimestamp() {
        return mTimestamp;
    }
}
