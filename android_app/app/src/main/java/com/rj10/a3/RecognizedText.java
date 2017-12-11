package com.rj10.a3;

import java.util.Date;

/**
 * Created by rjtang on 12/9/17.
 */

public class RecognizedText {
    private final String mText;
    private final Date mTimestamp;

    public RecognizedText(String text, Date timestamp) {
        mText = text;
        mTimestamp = timestamp;
    }

    public String getText() {
        return mText;
    }

    public Date getTimestamp() {
        return mTimestamp;
    }
}
