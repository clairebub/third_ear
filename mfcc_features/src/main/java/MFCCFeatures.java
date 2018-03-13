import be.tarsos.dsp.AudioDispatcher;
import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.io.jvm.AudioDispatcherFactory;
import be.tarsos.dsp.io.UniversalAudioInputStream;
import be.tarsos.dsp.io.TarsosDSPAudioFormat;
import be.tarsos.dsp.mfcc.MFCC;
import be.tarsos.dsp.io.PipedAudioStream;
import be.tarsos.dsp.io.TarsosDSPAudioInputStream;
import java.util.*;
import java.io.*;

//DURATION = 1
//N_MFCC= 40
//FREQUENCY_MIN = 20
//FREQUENCY_MAX = 11000

public class MFCCFeatures {
	//	private static int counter = 0;
	//@Test
	public List<float[]> Test() throws IOException {
		final String path = "data/7061-6-0-0.wav";
		final int SR = 44100;
		final int SAMPLE_SIZE_IN_BITS = 16;
		final int FREQUENCY_MAX = 11000;
		final int FREQUENCY_MIN = 20;

		final int numChannels = 2;
		final boolean isSigned = true;
		final boolean isBigEndian = false;


		printWaveHeader(path);

		InputStream inStream = new FileInputStream(path);
			TarsosDSPAudioFormat audioFormat = new TarsosDSPAudioFormat(
				SR, SAMPLE_SIZE_IN_BITS, numChannels, isSigned, isBigEndian);
		UniversalAudioInputStream audioInputStream = new UniversalAudioInputStream(
				inStream, audioFormat);
		audioInputStream.skip(44); // skip the size of wav header

		int samplesPerFrame = SR / 25;
		//int framesOverlap = samplesPerFrame / 4 * 3;
		int framesOverlap = 0; // deebug
		System.out.println("samplesPerFrame=" + samplesPerFrame);
		System.out.println("framesOverlap=" + framesOverlap);
		AudioDispatcher dispatcher = new AudioDispatcher(
				audioInputStream, samplesPerFrame, framesOverlap);
		System.out.println("dispatcher.format(): " + dispatcher.getFormat());

		final int N_MFCC = 40;
		int n_mels = samplesPerFrame / 10;
		System.out.println("n_mels=" + n_mels);
		final MFCC mfcc = new MFCC(
		    samplesPerFrame,
				SR,
				N_MFCC,
				n_mels,
				FREQUENCY_MIN,
				FREQUENCY_MAX);

		final List<float[]>mfccList = new ArrayList<>();
		float[] mfcc_avg = new float[N_MFCC];
		for(int i = 0; i < N_MFCC; i++) {
			mfcc_avg[i] = 0;
		}

		dispatcher.addAudioProcessor(new AudioProcessor() {
			int iFrames = 0;

			@Override
			public boolean process(AudioEvent audioEvent) {
				mfcc.process(audioEvent);
				float[] xx = mfcc.getMFCC();
				for(int i = 0; i < N_MFCC; i++) {
					mfcc_avg[i] += xx[i];
				}
				iFrames++;
				return true;
			}

			@Override
			public void processingFinished() {
				for (int i = 0; i < N_MFCC; i++) {
					mfcc_avg[i] /= iFrames;
					// System.out.println(mfcc_avg[i]);
				}
				System.out.println("iFrames=" + iFrames);
			}
		});

		dispatcher.run();
		System.out.println("seconds processed: " + dispatcher.secondsProcessed());
		return mfccList;
	}

	// WAVE header is 12 + 8 + 16 + 8 + ..
	private void printWaveHeader(String filename) throws IOException {
		// 0-3: RIFF
		// 4-7: chunk size
		// 8-11: fmt0
		// 12-15
		try (FileInputStream in = new FileInputStream(filename)) {
        byte[] bytes = new byte[4];

        // 0-3: RIFF
        if (in.read(bytes) < 0) {
            return;
        }
        printDescriptor("RIFF", bytes);

				// 4-7: filesize - 8
				if (in.read(bytes) < 0) {
						return;
				}
				printInt("filesize-8", bytes);

				// 8-11: WAVE
				if (in.read(bytes) < 0) {
            return;
        }
				printDescriptor("WAVE", bytes);

				// 12-15: fmt0
				if (in.read(bytes) < 0) {
						return;
				}
				printDescriptor("fmt0", bytes);

				// 16-19: should be size of the fmt chunk: 16
				if (in.read(bytes) < 0) {
						return;
				}
				printInt("fmtChunkSize", bytes);

				// skip the rest of the fmt chunk
				in.skip(16);


				// 36-39: "data"
				if (in.read(bytes) < 0) {
						return;
				}
				printDescriptor("data", bytes);

				// the data chunk size
				if (in.read(bytes) < 0) {
						return;
				}
				printInt("data chunk size", bytes);
			}
		}
/*
        // first subchunk will always be at byte 12
        // there is no other dependable constant
        in.skip(8);

        for (;;) {
            // read each chunk descriptor
            if (in.read(bytes) < 0) {
                break;
            }

            printDescriptor(bytes);

            // read chunk length
            if (in.read(bytes) < 0) {
                break;
            }

            // skip the length of this chunk
            // next bytes should be another descriptor or EOF
            in.skip(
                  (bytes[0] & 0xFF)
                | (bytes[1] & 0xFF) << 8
                | (bytes[2] & 0xFF) << 16
                | (bytes[3] & 0xFF) << 24
            );
        }

        System.out.println("end of file");
    } */

	private static void printInt(String name, byte[] bytes) throws IOException {
		int x = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).getInt();
		System.out.println(name + ": " + x);
	}

	private static void printDescriptor(String name, byte[] bytes) throws IOException {
	   String desc = new String(bytes, "US-ASCII");
		 System.out.println(name + ": " + desc);
	}

	public static void main(String[] args) throws IOException {
		System.out.println("Main thread name: " + Thread.currentThread().getName());
		MFCCFeatures a = new MFCCFeatures();
		a.Test();
	}
}
