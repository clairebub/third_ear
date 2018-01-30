package com.rj10.a3;

import java.util.Date;

public class RecognizedText {
    private final String mText;
    private final Date mTimestamp;
    private String wavFileName;

    public RecognizedText(String text, Date timestamp) {
        mText = text;
        mTimestamp = timestamp;
    }

    public String getWavFileName() {
        return wavFileName;
    }

    public void setWavFileName(String fileName) {
        wavFileName = fileName;
    }

    public String getText() {
        return mText;
    }

    public Date getTimestamp() {
        return mTimestamp;
    }
}
