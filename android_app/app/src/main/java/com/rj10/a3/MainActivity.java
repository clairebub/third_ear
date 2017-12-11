package com.rj10.a3;

import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.ComponentName;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.os.IBinder;
import android.os.Looper;
import android.speech.RecognizerIntent;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.DefaultItemAnimator;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.text.TextUtils;
import android.util.Log;
import android.util.TimingLogger;
import android.view.View;
import android.widget.Button;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

//
// An app that monitors the ambient sound and do speech detection and outputs the results
// on the screen.
// The UI is the following:
// 1. a button to start/stop the speech detection and indicate whether it's running
// 2. an output area to display the detected speech
//
public class MainActivity extends AppCompatActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {

    private static final String TAG = "HearItMainActivity";
    private final int PERMISSIONS_REQUEST_FOR_CLOUD = 1;

    private final int REQ_CODE_SPEECH_RECOG_LOCAL = 100;

    private TextView mStatus;
    private RadioGroup mRadioGroup;
    private RecyclerView mRecogTextListView;
    private List<RecognizedText> mRecogTextList = new ArrayList<>();
    private RecognizedTextsAdapter mRecogTextAdapter;

    private VoiceRecorder mVoiceRecorder;
    private SpeechApiService mSpeechApiService;
    int mRequestCount = 0;

    /** Defines callbacks for service binding, passed to bindService() */
    private final ServiceConnection mSpeechApiServiceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName className,
                                       IBinder service) {
            // We've bound to LocalService, cast the IBinder and get LocalService instance
            SpeechApiService.SpeechApiBinder binder = (SpeechApiService.SpeechApiBinder) service;
            mSpeechApiService = binder.getService();
            mSpeechApiService.setCallback(mSpeechApiServiceCallback);
            Log.d("SpeechApiService","onServiceConnected(): " + mSpeechApiService);
        }

        @Override
        public void onServiceDisconnected(ComponentName arg0) {
            mSpeechApiService = null;
            Log.d("SpeechApiService","onServiceDisconnected()" );
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate");

        setContentView(R.layout.activity_main);

        mStatus = (TextView) findViewById(R.id.status);
        mStatus.setHint("Version: 0.1c");
        Button onOffButton = (Button) findViewById(R.id.start);
        onOffButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mRequestCount++;
                handleOnClick((Button) v);
            }
        });

        mRadioGroup = (RadioGroup) findViewById(R.id.radioGroup);
        mRadioGroup.check(R.id.mode_streaming);
        mRadioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup radioGroup, int checkedId) {
                if (checkedId == R.id.mode_local) {
                    // cloud recognition runs in the background. need to make sure it is no
                    // longer running if mode is switching to local
                    stopSpeechRecCloud();
                }
            }
        });

        mRecogTextListView = (RecyclerView) findViewById(R.id.recog_texts);
        mRecogTextAdapter = new RecognizedTextsAdapter(mRecogTextList);
        RecyclerView.LayoutManager mLayoutManager = new LinearLayoutManager(getApplicationContext());
        ((LinearLayoutManager) mLayoutManager).setStackFromEnd(true);
        mRecogTextListView.setLayoutManager(mLayoutManager);
        mRecogTextListView.setItemAnimator(new DefaultItemAnimator());
        mRecogTextListView.setAdapter(mRecogTextAdapter);

        mRecogTextList.add(new RecognizedText("ready for speech recognition...", new Date()));
        mRecogTextAdapter.notifyDataSetChanged();
    }

    /**
     * Bind to the speech api service only when the activity is in use. You can do it at OnStart()
     * or at onResume()
     */
    @Override
    protected void onStart() {
        super.onStart();
        Log.d(TAG, "onStart");
        // Prepare Cloud Speech API
        bindService(new Intent(this, SpeechApiService.class),
                mSpeechApiServiceConnection, BIND_AUTO_CREATE);
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.d(TAG, "onPause");
    }

    @Override
    protected void onStop() {
        super.onStop();
        Log.d(TAG, "onStop");
        unbindService(mSpeechApiServiceConnection);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "onDestroy");
    }

    private void handleOnClick(Button button) {
        if (button.getText().toString().equalsIgnoreCase("start")) {
            startSpeechRec();
            button.setText("Stop");
        } else {
            stopSpeechRec();
            button.setText("Start");
        }
    }

    private void startSpeechRec() {
        int radioButtonID = mRadioGroup.getCheckedRadioButtonId();
        switch (radioButtonID) {
            case R.id.mode_local:
                startSpeechRecLocal();
                break;
            case R.id.mode_streaming:
                startSpeechRecCloud();
                break;
            default:
                Log.d(TAG,"unexpected radioButtonId: " + radioButtonID);
                break;
        }
    }

    private void stopSpeechRec() {
        int radioButtonID = mRadioGroup.getCheckedRadioButtonId();
        switch (radioButtonID) {
            case R.id.mode_streaming:
                stopSpeechRecCloud();
                break;
            default:
                Log.d(TAG, "unexpected radioButtonId: " + radioButtonID);
                break;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode) {
            case REQ_CODE_SPEECH_RECOG_LOCAL: {
                if (resultCode == RESULT_OK && null != data) {
                    ArrayList<String> result = data
                            .getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
                    Log.d(TAG, result.get(0));
                }
                break;
            }
            default:
                Log.d(TAG, "unexpected request code: " + requestCode);
                break;
        }
    }

    private void startSpeechRecLocal() {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());
        intent.putExtra(
                RecognizerIntent.EXTRA_PROMPT,
                "Starting speech recognition...");
        try {
            startActivityForResult(intent, REQ_CODE_SPEECH_RECOG_LOCAL);
        } catch (ActivityNotFoundException a) {
            Log.d(TAG, a.toString());
        }
    }

    private void startSpeechRecCloud() {
        String[] permissions = new String[] {
                Manifest.permission.INTERNET,
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.READ_PHONE_STATE
        };
        if (!checkPermissions(permissions, PERMISSIONS_REQUEST_FOR_CLOUD)) {
            return;
        }

        if (mVoiceRecorder != null) {
            mVoiceRecorder.stop();
        }
        mVoiceRecorder = new VoiceRecorder();
        mVoiceRecorder.start(mVoiceCallback);
    }


    private void stopSpeechRecCloud() {
        Log.d(TAG, "stopSpeechRecCloud: " + mVoiceRecorder);
        if (mVoiceRecorder != null) {
            mVoiceRecorder.stop();
            mVoiceRecorder = null;
        }
    }

    private final SpeechApiService.Callback mSpeechApiServiceCallback =
            new SpeechApiService.Callback() {
        @Override
        public void onSpeechRecognized(final String text, final boolean isFinal) {
            if (isFinal) {
                // TODO:
            }
            if (text != null && !TextUtils.isEmpty(text)) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        //if(isFinal) {
                            mRecogTextList.add(new RecognizedText(text, new Date()));
                            Log.d(TAG, "isFinal=" + isFinal + ", got text: " + text);
                            mRecogTextAdapter.notifyDataSetChanged();
                        //}
                        mRecogTextListView.scrollToPosition(mRecogTextList.size()-1);
                    }
                });
            }
        }
    };

    //
    // Callback from voice recorder to handle start/ongoing/end of voice input.
    // For cloud based speech recog streaming Api only.
    // The callbacks of onVoice() are called from a separate thread.
    //
    private final VoiceRecorder.Callback mVoiceCallback = new VoiceRecorder.Callback() {
        @Override
        public void onVoiceStart(int sampleRate) {
            Log.d(TAG, "VoiceRecorder.onVoiceStart(): " + (Looper.myLooper() == Looper.getMainLooper()));
            if (mSpeechApiService == null) {
                Log.d(TAG, "mSpeechApiService is null onVoiceStart");
                return;
            }
            mSpeechApiService.startRecognizing(sampleRate);
        }

        @Override
        public void onVoice(byte[] data, int size) {
            // Log.d(TAG,"VoiceRecorder.onVoice(): " + (Looper.myLooper() == Looper.getMainLooper()));
            if (mSpeechApiService == null) {
                Log.d(TAG, "mSpeechApiService is null onVoice");
                return;
            }
            mSpeechApiService.recognize(data, size);
        }

        @Override
        public void onVoiceEnd() {
            Log.d(TAG,"VoiceRecorder.onVoiceEnd(): " + (Looper.myLooper() == Looper.getMainLooper()));
            if (mSpeechApiService != null) {
                Log.d(TAG,"mSpeechApiService is null onVoiceEnd");
                return;
            }
            mSpeechApiService.finishRecognizing();
        }

        @Override
        public void deebug(String msg) {
            Log.d(TAG,"VoiceRecorder.deebug(): " + msg + ", " + (Looper.myLooper() == Looper.getMainLooper()));
            //showStatus(msg);
        }
    };

    /**
     * Returns whether it has the permission, start the permission request flow asynchchronously
     * if it deosn't have the permission.
     * @param permissions
     * @param requestCode
     * @return true if it has the permission.
     */
    private boolean checkPermissions(String[] permissions, int requestCode) {
        List<String> permissionsMissing = new ArrayList<>();
        for (String p: permissions) {
            if (ContextCompat.checkSelfPermission(this, p)
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsMissing.add(p);
            }
        }
        if (permissionsMissing.isEmpty()) {
            return true;
        }
        String msg = "";
        for (String p: permissionsMissing) {
            if (!msg.isEmpty()) {
                msg = msg = ", ";
            }
            msg += p;
        }
        Log.d(TAG, "requesting permission for " + msg);
        String[] p2 = new String[permissionsMissing.size()];
        for(int i = 0; i < permissionsMissing.size(); i++) {
            p2[i] = permissionsMissing.get(i);
        }
        ActivityCompat.requestPermissions(this,
                p2,
                requestCode);
        return false;
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == PERMISSIONS_REQUEST_FOR_CLOUD) {
            for (int i = 0; i < grantResults.length; i++) {
                if (grantResults[i] != PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Missing permission: " + permissions[i],
                            Toast.LENGTH_SHORT).show();
                }
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    void showStatus(final CharSequence msg) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mStatus.setText(msg);
            }
        });
    }
}
