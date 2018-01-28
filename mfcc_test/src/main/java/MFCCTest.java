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
public class MFCCTest {

//	private static int counter = 0;

	//@Test
	public List<float[]> Test() throws FileNotFoundException {
		final String path = "data/test.wav";
		final int sampleRate = 16000;
		final int bufferSize = 1024;
		final int bufferOverlap = 128;
		final int samplesPerFrame = -1;
		final int amountOfCepstrumCoef = 20;
		final int amountOfMelFilters = 50;
		final int lowerFilterFreq = 25;
		final int upperFilterFreq = 200000;

		InputStream inStream = new FileInputStream(path);
			TarsosDSPAudioFormat audioFormat = new TarsosDSPAudioFormat(
				sampleRate, 16, 1, true, true);
		UniversalAudioInputStream audioInputStream = new UniversalAudioInputStream(
				inStream, audioFormat);
		AudioDispatcher dispatcher = new AudioDispatcher(
				audioInputStream, bufferSize, bufferOverlap);
		System.out.println("dispatcher.format(): " + dispatcher.getFormat());
		final List<float[]>mfccList = new ArrayList<>();
		final MFCC mfcc = new MFCC(
		    bufferSize,
				sampleRate,
				amountOfCepstrumCoef,
				amountOfMelFilters,
				lowerFilterFreq,
				upperFilterFreq);
		dispatcher.addAudioProcessor(mfcc);
		dispatcher.addAudioProcessor(new AudioProcessor() {
			@Override
			public boolean process(AudioEvent audioEvent) {
				mfccList.add(mfcc.getMFCC());
				return true;
			}
			@Override
			public void processingFinished() {}
		});

		dispatcher.run();
		return mfccList;
	}

	public static void main(String[] args) throws FileNotFoundException {
		MFCCTest a = new MFCCTest();
		List<float[]> mfcc = a.Test();
		System.out.println("mfcc size: " + mfcc.size());
		System.out.println("mfcc values: ");
		for (float[] x: mfcc) {
			System.out.println(Arrays.toString(x));
		}
	}
}
