package com.rj10.a3;

import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.os.Handler;
import android.speech.RecognizerIntent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private final int REQ_CODE_SPEECH_INPUT = 100;

    Button buttonOnOff = null;
    SpeechResultReceiver speechResultReceiver = null;
    int clickCount = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        speechResultReceiver = new SpeechResultReceiver(new Handler());

        buttonOnOff = (Button) findViewById(R.id.buttonOnOff);
        buttonOnOff.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                clickCount += 1;
                String msg = String.format("clicked %d", clickCount);
                Log.i("Deebug", msg);
                Toast.makeText(getApplicationContext(), msg, Toast.LENGTH_LONG).show();
                startSpeedRec();
            }
        });
    }

    void startSpeedRec() {
        Intent intent = new Intent(this, SpeechService.class);
        intent.putExtra("receiver", speechResultReceiver);
        intent.putExtra("sender", "stem main");
        startService(intent);

    }
/*
    void startSpeedRec() {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());
        intent.putExtra(RecognizerIntent.EXTRA_PROMPT,
                "Hi speak something");
        try {
            startActivityForResult(intent, REQ_CODE_SPEECH_INPUT);
        } catch (ActivityNotFoundException a) {
            Log.w("Deebug", a);
        }
    }
*/
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode) {
            case REQ_CODE_SPEECH_INPUT: {
                if (resultCode == RESULT_OK && null != data) {

                    ArrayList<String> result = data
                            .getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
                   Log.w("Deebug", result.get(0));
                }
                break;
            }

        }
    }
}
