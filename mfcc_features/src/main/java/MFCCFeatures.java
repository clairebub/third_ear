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
	public List<float[]> Test() throws FileNotFoundException {
		final String path = "data/7061-6-0-0.wav";
		final int SR = 44100;
		final int SAMPLE_SIZE_IN_BITS = 16;
		final int FREQUENCY_MAX = 11000;
		final int FREQUENCY_MIN = 20;

		final int numChannels = 2;
		final boolean isSigned = true;
		final boolean isBigEndian = false;

		InputStream inStream = new FileInputStream(path);
			TarsosDSPAudioFormat audioFormat = new TarsosDSPAudioFormat(
				SR, SAMPLE_SIZE_IN_BITS, numChannels, isSigned, isBigEndian);
		UniversalAudioInputStream audioInputStream = new UniversalAudioInputStream(
				inStream, audioFormat);

		int samplesPerFrame = SR / 25;
		int framesOverlap = samplesPerFrame / 4 * 3;
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
					System.out.println(mfcc_avg[i]);
				}
			}
		});

		dispatcher.run();
		System.out.println("seconds processed: " + dispatcher.secondsProcessed());
		return mfccList;
	}

	public static void main(String[] args) throws FileNotFoundException {
		for (String arg: args) {
			System.out.println("arg: " + arg);
		}
		MFCCFeatures a = new MFCCFeatures();
		List<float[]> mfcc = a.Test();
//		System.out.println():
		System.out.println("mfcc size: " + mfcc.size());
		System.out.println("mfcc values: ");
		for (float[] x: mfcc) {
			//System.out.println("deebug: " + x.length);
			//System.out.println(Arrays.toString(x));
		}
	}
}
