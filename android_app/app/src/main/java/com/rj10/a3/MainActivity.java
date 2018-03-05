package com.rj10.a3;

import android.Manifest;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
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
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import be.tarsos.dsp.AudioDispatcher;
import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.io.android.AudioDispatcherFactory;
import be.tarsos.dsp.pitch.PitchDetectionHandler;
import be.tarsos.dsp.pitch.PitchDetectionResult;
import be.tarsos.dsp.pitch.PitchProcessor;

//
// An app that monitors the ambient sound and do speech detection and outputs the results
// on the screen.
// The UI is the following:
// 1. a button to start/stop the speech detection and indicate whether it's running
// 2. an output area to display the detected speech
//
public class MainActivity extends AppCompatActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {

    private static final String TAG = "3rdEarMainActivity";
    private final int PERMISSIONS_REQUEST_FOR_CLOUD = 1;

    private final int REQ_CODE_SPEECH_RECOG_LOCAL = 100;

    private TextView mStatusTextView;
    private Button mStartButton;
    private Button mStopButton;
    private RecyclerView mRecogTextListView;
    private List<RecognizedText> mRecogTextList = new ArrayList<>();
    private RecognizedTextsAdapter mRecogTextAdapter;

    private SoundRecorder mVoiceRecorder;
    private SpeechApiService mSpeechApiService;

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

        mStatusTextView = (TextView) findViewById(R.id.statusTextView);
        mStatusTextView.setHint("Version: 0.3");

        mStartButton = (Button) findViewById(R.id.startButton);
        mStartButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                handleStartButtonClick((Button) v);
            }
        });
        mStartButton.setEnabled(true);

        mStopButton = (Button) findViewById(R.id.stopButton);
        mStopButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                handleStopButtonClick((Button) v);
            }
        });
        mStopButton.setEnabled(false);

        mRecogTextListView = (RecyclerView) findViewById(R.id.recog_texts);
        RecycleViewClickListener listener = new RecycleViewClickListener(this);
        mRecogTextAdapter = new RecognizedTextsAdapter(mRecogTextList, listener);
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

    private void handleStartButtonClick(Button button) {
        mStartButton.setEnabled(false);
        mStopButton.setEnabled(true);
        startSpeechRec();
    }

    private void handleStopButtonClick(Button button) {
        mStopButton.setEnabled(false);
        mStartButton.setEnabled(true);
        stopSpeechRec();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode) {
            case REQ_CODE_SPEECH_RECOG_LOCAL: {
                if (resultCode == RESULT_OK && null != data) {
                    ArrayList<String> result = data
                            .getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
                    String recogText = result.get(0);
                    Log.d(TAG, recogText);
                    mRecogTextList.add(new RecognizedText(recogText, new Date()));
                    mRecogTextAdapter.notifyDataSetChanged();
                }
                break;
            }
            default:
                Log.d(TAG, "unexpected request code: " + requestCode);
                break;
        }
    }
/*
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
*/
    private void startSpeechRec() {
        String[] permissions = new String[] {
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.READ_PHONE_STATE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
        };
        if (!checkPermissions(permissions, PERMISSIONS_REQUEST_FOR_CLOUD)) {
            return;
        }

        if (mVoiceRecorder != null) {
            mVoiceRecorder.stop();
        }
        mVoiceRecorder = new SoundRecorder();
        mVoiceRecorder.start(mVoiceCallback);
    }


    private void stopSpeechRec() {
        Log.d(TAG, "stopSpeechRecStreaming: " + mVoiceRecorder);
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
    private final SoundRecorder.Callback mVoiceCallback = new SoundRecorder.Callback() {
        private int mSampleRate = -1;
        @Override
        public void onVoiceStart(int sampleRate) {
            mSampleRate = sampleRate;
            Log.d(TAG, "SoundRecorder.onVoiceStart(): " + (Looper.myLooper() == Looper.getMainLooper()));
            if (mSpeechApiService == null) {
                Log.d(TAG, "mSpeechApiService is null onVoiceStart");
                return;
            }
            mSpeechApiService.startRecognizing(sampleRate);
        }

        @Override
        public void onVoice(byte[] data, int size) {
            // Log.d(TAG,"SoundRecorder.onVoice(): " + (Looper.myLooper() == Looper.getMainLooper()));
            if (mSpeechApiService == null) {
                Log.d(TAG, "mSpeechApiService is null onVoice");
                return;
            }
            if (mSampleRate < 0) {
                Log.e(TAG, "sampleRate is -1 when onVoice");
                return;
            }
            /*
            // get the frames from the audio buffer
            double[][] frames = getAudioFrames(data, size);

            int nnumberofFilters = 24;
            int nlifteringCoefficient = 22;
            boolean oisLifteringEnabled = true;
            boolean oisZeroThCepstralCoefficientCalculated = true;
            int nnumberOfMFCCParameters = 12; //without considering 0-th
            double dsamplingFrequency = 8000.0;
            int nFFTLength = 512;

            MFCC mfcc = new MFCC(nnumberOfMFCCParameters,
                    dsamplingFrequency,
                    nnumberofFilters,
                    nFFTLength,
                    oisLifteringEnabled,
                    nlifteringCoefficient,
                    oisZeroThCepstralCoefficientCalculated);
            for (int i = 0; i < frames.length; i++) {
                double[] mfccFeatures = mfcc.getParameters(frames[i]);
                Log.d(TAG, String.format("mfcc feature length: " + mfccFeatures.length));
            } */
           // mSpeechApiService.recognize(data, size);
        }

        @Override
        public void onVoiceEnd() {
            Log.d(TAG,"SoundRecorder.onVoiceEnd(): " + (Looper.myLooper() == Looper.getMainLooper()));
            if (mSpeechApiService != null) {
                Log.d(TAG,"mSpeechApiService is null onVoiceEnd");
                return;
            }
            mSpeechApiService.finishRecognizing();
        }

        @Override
        public void deebug(String msg) {
            Log.d(TAG,"SoundRecorder.deebug(): " + msg + ", " + (Looper.myLooper() == Looper.getMainLooper()));
            //showStatus(msg);
        }

        @Override
        public void setWaveFile(final String waveFilePath) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    String fileName = waveFilePath;
                    mRecogTextList.add(new RecognizedText(fileName, new Date()));
                    mRecogTextAdapter.notifyDataSetChanged();
                    mRecogTextListView.scrollToPosition(mRecogTextList.size()-1);
                }
            });

        }

        private double[][] getAudioFrames(byte[] data, int size) {
            int totalSamples = size / 2;
            int frameDuration = 40; // 40 ms
            int samplesPerFrame = (int) (mSampleRate / 1000 * frameDuration);
            int stride_length = samplesPerFrame / 4;
            int totalFrames = totalSamples / samplesPerFrame;
            Log.d(TAG, String.format("total frames=%d, samplesPerFrame=%d, stride_length=%d", totalFrames, samplesPerFrame, stride_length));

            ByteBuffer byteBuffer = ByteBuffer.wrap(data, 0, size);
            if (ByteOrder.nativeOrder().equals(ByteOrder.BIG_ENDIAN)) {
                byteBuffer.order(ByteOrder.BIG_ENDIAN); // assume it's little endian
                Log.d(TAG, "big endian");
            } else {
                byteBuffer.order(ByteOrder.LITTLE_ENDIAN); // assume it's little endian
                Log.d(TAG, "little endian");
            }

            double[] pcmValues = new double[totalSamples];
            int count[] = new int[100];
            for (int i = 0; i < totalSamples; i++) {
                pcmValues[i] = 1.0 * byteBuffer.getInt(i) / Integer.MAX_VALUE;
                int x = (int) (100 * (pcmValues[i] + 1)/2.0) - 1;
                if (x < 0) {
                    x = 0;
                }
                count[x] += 1;
            }
            String s = "";
            for (int i = 0; i < 100; i++) {
                s += ", " + count[i];
            }
            Log.d(TAG, String.format("count=%s", s));

            double[][] frames = new double[totalFrames][samplesPerFrame];
            for (int i = 0; i < totalFrames; i++) {
                int offset = i * samplesPerFrame;
                for (int j = 0; j < samplesPerFrame; j++) {
                    int x = offset + j;
                    if (x >= totalSamples) {
                        frames[i][j] = 0;
                    } else {
                        frames[i][j] = pcmValues[x];
                    }
                }
            }
            return frames;
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
                mStatusTextView.setText(msg);
            }
        });
    }

    private void tryTarsos() {
        AudioDispatcher dispatcher = AudioDispatcherFactory.fromDefaultMicrophone(22050,1024,0);

        PitchDetectionHandler pdh = new PitchDetectionHandler() {
            @Override
            public void handlePitch(PitchDetectionResult result, AudioEvent e) {
                final float pitchInHz = result.getPitch();
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        TextView text = (TextView) findViewById(R.id.statusTextView);
                        text.setText("" + pitchInHz);
                    }
                });
            }
        };
        AudioProcessor p = new PitchProcessor(PitchProcessor.PitchEstimationAlgorithm.FFT_YIN, 22050, 1024, pdh);
        dispatcher.addAudioProcessor(p);
        new Thread(dispatcher,"Audio Dispatcher").start();
    }

    private void playWavFile() {
        MediaPlayer mediaPlayer = MediaPlayer.create(getApplicationContext(), R.raw.applause_y);
        mediaPlayer.start();
    }

    class RecycleViewClickListener implements RecognizedTextsAdapter.ClickListener {
        private final Context mContext;
        private MediaPlayer mMediaPlayer;

        public RecycleViewClickListener(Context context) {
            mContext = context;
        }
        public void onClick(String waveFileName) {
            try {
                if (mMediaPlayer != null) {
                    mMediaPlayer.release();
                    mMediaPlayer = null;
                }
                mMediaPlayer = new MediaPlayer();
                mMediaPlayer.setDataSource(waveFileName);
                mMediaPlayer.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
                    @Override
                    public void onPrepared(MediaPlayer player) {
                        player.start();
                    }
                });
                mMediaPlayer.prepareAsync();
            } catch (IOException ex) {
                Toast.makeText(mContext, waveFileName + " not found", Toast.LENGTH_SHORT).show();
                Log.e(TAG, ex.toString());
            }
        }
    }
}
