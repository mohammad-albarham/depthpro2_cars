//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import AppKit

extension NSVisualEffectView {
    fileprivate func panel() {
        material = .hudWindow
        blendingMode = .withinWindow
        state = .active
        appearance = NSAppearance(named: .darkAqua)
        wantsLayer = true
        layer?.cornerRadius = 8
        translatesAutoresizingMaskIntoConstraints = false
    }
}

final class Pill: NSVisualEffectView {
    private let label = NSTextField(labelWithString: "")
    private let size: CGFloat
    private let weight: NSFont.Weight

    init(size: CGFloat = 12, weight: NSFont.Weight = .medium) {
        (self.size, self.weight) = (size, weight)
        super.init(frame: .zero)
        panel()
        label.translatesAutoresizingMaskIntoConstraints = false
        addSubview(label)
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            label.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10),
            label.topAnchor.constraint(equalTo: topAnchor, constant: 6),
            label.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -6),
        ])
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    func show(_ primary: String, _ secondary: String? = nil) {
        let text = NSMutableAttributedString(
            string: primary,
            attributes: [.font: NSFont.monospacedDigitSystemFont(ofSize: size, weight: weight), .foregroundColor: NSColor.white])
        if let secondary {
            text.append(
                NSAttributedString(
                    string: "\n" + secondary,
                    attributes: [
                        .font: NSFont.monospacedSystemFont(ofSize: size - 1, weight: .regular),
                        .foregroundColor: NSColor.white.withAlphaComponent(0.6),
                    ]))
        }
        label.attributedStringValue = text
    }
}

/// A pop-up menu on a panel like the pills'.
final class Picker: NSVisualEffectView {
    var onPick: ((Int) -> Void)?
    private let button = NSPopUpButton(frame: .zero, pullsDown: false)

    init() {
        super.init(frame: .zero)
        panel()
        isHidden = true
        button.isBordered = false
        button.font = .systemFont(ofSize: 12, weight: .medium)
        button.refusesFirstResponder = true
        button.target = self
        button.action = #selector(picked)
        button.translatesAutoresizingMaskIntoConstraints = false
        addSubview(button)
        NSLayoutConstraint.activate([
            button.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 6),
            button.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -6),
            button.topAnchor.constraint(equalTo: topAnchor, constant: 6),
            button.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -6),
        ])
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    func show(_ titles: [String], selected: Int?) {
        button.removeAllItems()
        for title in titles { button.menu?.addItem(withTitle: title, action: nil, keyEquivalent: "") }
        button.selectItem(at: selected ?? -1)
        isHidden = titles.isEmpty
    }

    @objc private func picked() { onPick?(button.indexOfSelectedItem) }
}

final class ColorBar: NSVisualEffectView {
    private let gradient = CAGradientLayer()
    private let labels = (0..<3).map { _ in NSTextField(labelWithString: "") }

    init() {
        super.init(frame: .zero)
        panel()
        let steps = 32
        gradient.colors = (0...steps).map { step -> CGColor in
            let c = turbo(1 - Float(step) / Float(steps))
            return CGColor(red: CGFloat(c.x), green: CGFloat(c.y), blue: CGFloat(c.z), alpha: 1)
        }
        (gradient.startPoint, gradient.endPoint) = (CGPoint(x: 0, y: 0.5), CGPoint(x: 1, y: 0.5))
        gradient.cornerRadius = 3
        layer?.addSublayer(gradient)
        for label in labels {
            label.font = .monospacedDigitSystemFont(ofSize: 10, weight: .medium)
            label.textColor = NSColor.white.withAlphaComponent(0.75)
        }
        let row = NSStackView(views: labels)
        row.distribution = .equalSpacing
        row.translatesAutoresizingMaskIntoConstraints = false
        addSubview(row)
        NSLayoutConstraint.activate([
            heightAnchor.constraint(equalToConstant: 36),
            row.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            row.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10),
            row.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -5),
        ])
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    func show(_ range: ClosedRange<Double>?) {
        isHidden = range == nil
        guard let range else { return }
        let middle = (range.lowerBound * range.upperBound).squareRoot()
        for (label, meters) in zip(labels, [range.lowerBound, middle, range.upperBound]) {
            label.stringValue = meters.formatted(.number.precision(.significantDigits(1...3)).grouping(.never))
        }
        labels[0].stringValue += " m"
    }

    override func layout() {
        super.layout()
        gradient.frame = CGRect(x: 12, y: bounds.height - 15, width: bounds.width - 24, height: 6)
    }
}

// Turbo colormap, polynomial approximation by Ruofei Du of Anton Mikhailov's design:
// https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
// Copyright 2019 Google LLC. SPDX-License-Identifier: Apache-2.0
func turbo(_ x: Float) -> SIMD3<Float> {
    let v4 = SIMD4<Float>(1, x, x * x, x * x * x)
    let v2 = SIMD2<Float>(v4.z * v4.z, v4.w * v4.z)
    let r = (v4 * SIMD4(0.13572138, 4.61539260, -42.66032258, 132.13108234)).sum() + (v2 * SIMD2(-152.94239396, 59.28637943)).sum()
    let g = (v4 * SIMD4(0.09140261, 2.19418839, 4.84296658, -14.18503333)).sum() + (v2 * SIMD2(4.27729857, 2.82956604)).sum()
    let b = (v4 * SIMD4(0.10667330, 12.64194608, -60.58204836, 110.36276771)).sum() + (v2 * SIMD2(-89.90310912, 27.34824973)).sum()
    return SIMD3(r, g, b).clamped(lowerBound: SIMD3(repeating: 0), upperBound: SIMD3(repeating: 1))
}
